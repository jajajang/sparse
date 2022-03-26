from functools import wraps
from typing import List, Tuple
import mosek
import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
import cvxpy as cp

from cvxpy.atoms import bmat, reshape, trace, upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.constraints.constraint import Constraint


def loss_fn(X, Y, beta,T0):
    return cp.norm2(X @ beta - Y) ** 2 / T0

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd, T0):
    return loss_fn(X, Y, beta,T0) + lambd * regularizer(beta)

#Changed loss fn to average definition
#Thresholding

d=20
s=2
delta=0.05
opt=1/np.sqrt(d)

#create action set that satisfies M2 << Cmin^-1
A=np.zeros((d,d))
A[0][0]=opt
for i in range(1,d):
    A[i][0]=1
    A[i][i]=opt

#Set1: Our Settings
mu=cp.Variable(d, pos=True)
X=A.T@ cp.diag(mu) @A

T = Variable((d, d), symmetric=True)
M = bmat([[X, np.eye(d)],
          [np.eye(d), T]])
constraints = [M >> 0, cp.sum(mu)==1]
objective = cp.Minimize(cp.max(cp.diag(T)))

prob=cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK,verbose=True)

#results of set1
M2=prob.value
print('Set1 Optimization Finished\n')
print('Mu')
print(mu.value)
mu_dist=mu.value/np.sum(mu.value)
print('M2')
print(prob.value)
Q=A.T@ np.diag(mu_dist) @A
Q_inv=LA.inv(Q)


#Set2: Usual Botao hao setting optimization
mu_hao=cp.Variable(d, pos=True)
X_hao=A.T@ cp.diag(mu_hao) @A

constraints_hao = [cp.sum(mu_hao)==1]
prob_hao = cp.Problem(cp.Maximize(cp.lambda_min(X_hao)), constraints_hao)
prob_hao.solve(solver=cp.MOSEK, verbose=True)

print('Set2 Optimization Finished\n')
print('Mu_hao')
print(mu_hao.value)
dist_hao=mu_hao.value/np.sum(mu_hao.value)
print('Cmin')
Cmin=prob_hao.value
print(prob_hao.value)



#Experiment setup - variables
T0=0
repeative=30
howlong=20
wholelength=repeative*howlong
sigma=0.01


hist_true=np.zeros((wholelength,d))             #history of true theta
hist_esti=np.zeros((wholelength,d))             #history of the estimated theta by our method
errors=np.zeros(howlong)                        #l2 error between true and our estimation
choice_errors=np.zeros(howlong)                 #action error

hist_esti_hao=np.zeros((wholelength,d))         #history of the estimated theta by Hao's method
errors_hao=np.zeros(howlong)                    #l2 error between true and Hao's estimation
choice_errors_h=np.zeros(howlong)               #action error of Hao

for i in range(0,howlong):
    T0=T0+1000
    print('=====Repetition for total time %d=====' % (T0))
    for rep in range(0,repeative):
        theta = np.zeros(d)
        theta[0]=-opt
        theta[np.random.choice(d-1)+1]=1
        a_true=A[np.argmax(A@theta)]
        S=np.max(A@theta)

        theta_raw = 0
        threshold = np.sqrt((S**2 + sigma**2)*M2/T0*np.log(d/delta))
        hist_true[i*repeative+rep]=theta
        X_hist=np.zeros((d,d))
        for t in range(0,T0):
            act_t=A[np.random.choice(d, p=mu_dist)]
            r=theta@act_t+np.random.normal(0,sigma)
            theta_raw=theta_raw+r*act_t/T0                  #add reward*action/T0
            #theta_raw=theta_raw+r*act_t
            #X_hist=X_hist+np.outer(act_t, act_t)
        theta_hat_raw=Q_inv@theta_raw
        theta_hat=(theta_hat_raw>threshold)*theta_hat_raw
        #theta_hat=LA.inv(X_hist)@theta_raw
        hist_esti[i*repeative+rep]=theta_hat
        errors[i]+=LA.norm(theta-theta_hat)
        a_hat=A[np.argmax(A@theta_hat)]
        choice_errors[i]+=LA.norm(a_true-a_hat)


        hist_b=np.zeros((T0, d))                        #temporary history for the action of Hao's method, since it computes LASSO optimization
        r_b=np.zeros(T0)                                #temporary history for the reward
        for t in range(0,T0):
            act_h_t = A[np.random.choice(d, p=dist_hao)]
            hist_b[t]=act_h_t
            r_b[t] = theta @ act_h_t + np.random.normal(0, sigma)
        beta = cp.Variable(d)
        lambd_b = 4 * np.sqrt(np.log(d) / T0)
        lassosol = cp.Problem(cp.Minimize(objective_fn(hist_b, r_b, beta, lambd_b, T0)))
        lassosol.solve()
        beta_hat = beta.value
        hist_esti_hao[i*repeative+rep]=beta_hat
        errors_hao[i]+=LA.norm(theta-beta_hat)
        a_hat_h=A[np.argmax(A@beta_hat)]
        choice_errors_h[i]+=LA.norm(a_true-a_hat_h)

print(errors)
print(errors_hao)
print(choice_errors)
print(choice_errors_h)