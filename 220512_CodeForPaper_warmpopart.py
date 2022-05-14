from functools import wraps
from typing import List, Tuple
import mosek
import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
import cvxpy as cp
import argparse
from my_catoni import *

from cvxpy.atoms import bmat, reshape, trace, upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.constraints.constraint import Constraint


parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, help="dimension of the action vector", default=50)
parser.add_argument("--swap", type=bool, help='Whether we should swap the strategy between Haos exploration and ours', default=False)
parser.add_argument("--inc",type=int, help='Incremental scale of the experiment', default=1000)
parser.add_argument("--repeat",type=int, help='How many times it will repeat for each experiment setting', default=30)
parser.add_argument("--howlong",type=int, help='Maximum multiple of T_min for the experiment. For example, howlong=20 and inc=2000 means maximum 40000 rounds', default=20)
parser.add_argument("--T_min",type=int, help='Minimum length of experiment to examine', default=0)
parser.add_argument("--sigma",type=float, help='Variance', default=1)
parser.add_argument("--delta",type=float, help='Error probability in bandit', default=0.05)

args = parser.parse_args()


def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y) ** 2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


d=args.d
s=2
delta=args.delta

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


if args.swap:
    p_first=dist_hao
    p_second=mu_dist
else:
    p_first=mu_dist
    p_second=dist_hao


#Experiment setup - variables
T0=args.T_min
repeative=args.repeat
howlong=args.howlong
wholelength=repeative*howlong
sigma=args.sigma


hist_true=np.zeros((wholelength,d))             #history of true theta
hist_esti=np.zeros((wholelength,d))             #history of the estimated theta by our method
hist_esti_raw=np.zeros((wholelength,d))
all_error=np.zeros(wholelength)
errors=np.zeros(howlong)                        #l2 error between true and our estimation
choice_errors=np.zeros(howlong)                 #action error

hist_esti_hao=np.zeros((wholelength,d))         #history of the estimated theta by Hao's method
errors_hao=np.zeros(howlong)                    #l2 error between true and Hao's estimation
all_error_hao=np.zeros(wholelength)
choice_errors_h=np.zeros(howlong)               #action error of Hao

vari_our=np.zeros(howlong)
vari_hao=np.zeros(howlong)

for i in range(0,howlong):
    T0=T0+args.inc
    print('=====Repetition for total time %d=====' % (T0))
    for rep in range(0,repeative):
        #setting theta - changes over each experiment
        theta = np.zeros(d)
        theta[0]=-1
        theta[np.random.choice(d-1)+1]=1

        #same setting from here - threshold and catoni
        a_true=A[np.argmax(A@theta)]
        S=np.max(np.abs(A@theta))
        vari_0 = (S**2 + sigma**2)*M2
        vari_1 = (2*sigma**2)*M2
        threshold_0 = width_catoni(T0/2,d,delta,vari_0)
        threshold_1 = width_catoni(T0/2,d,delta,vari_1)
        hist_true[i*repeative+rep]=theta
        X_hist=np.zeros((T0,d))
        for t in range(0,T0//2):
            act_t=A[np.random.choice(d, p=p_first)]
            r=theta@act_t+np.random.normal(0,sigma)
            X_hist[t]=r*(Q_inv@act_t)
        theta_hat_warm=catoni_esti(X_hist[:T0//2],delta,vari_0)
        theta_0 = (np.abs(theta_hat_warm) > threshold_0) * theta_hat_warm
        for t in range(0, T0 // 2):
            act_t = A[np.random.choice(d, p=p_first)]
            r = theta @ act_t + np.random.normal(0, sigma)
            X_hist[t+T0//2] = theta_0 + (r-theta_0 @ act_t) * (Q_inv @ act_t)
        theta_hat_raw=catoni_esti(X_hist[T0//2:T0],delta,vari_1)
        hist_esti_raw[i*repeative+rep]=theta_hat_raw
        theta_hat=(np.abs(theta_hat_raw)>threshold_1)*theta_hat_raw
        #theta_hat=LA.inv(X_hist)@theta_raw
        hist_esti[i*repeative+rep]=theta_hat
        all_error[i*repeative+rep]=LA.norm(theta-theta_hat,1)
        errors[i]+=LA.norm(theta-theta_hat,1)/repeative
        a_hat=A[np.argmax(A@theta_hat)]
        choice_errors[i]+=LA.norm(a_true-a_hat)/repeative


        hist_b=np.zeros((T0, d))                        #temporary history for the action of Hao's method, since it computes LASSO optimization
        r_b=np.zeros(T0)                                #temporary history for the reward
        for t in range(0,T0):
            act_h_t = A[np.random.choice(d, p=p_second)]
            hist_b[t]=act_h_t
            r_b[t] = theta @ act_h_t + np.random.normal(0, sigma)
        beta = cp.Variable(d)
        lambd_b = 4 * sigma*np.sqrt(np.log(d) * T0)
        lassosol = cp.Problem(cp.Minimize(objective_fn(hist_b, r_b, beta, lambd_b)))
        lassosol.solve()
        beta_hat = beta.value
        hist_esti_hao[i*repeative+rep]=beta_hat
        errors_hao[i]+=LA.norm(theta-beta_hat,1)/repeative
        all_error_hao[i*repeative+rep]=LA.norm(theta-beta_hat,1)
        a_hat_h=A[np.argmax(A@beta_hat)]
        choice_errors_h[i]+=LA.norm(theta-a_hat_h)
    vari_our[i]=np.var(all_error[repeative*i:repeative*(i+1)-1])
    vari_hao[i]=np.var(all_error_hao[repeative*i:repeative*(i+1)-1])

import matplotlib.pyplot as plt
timeline=np.linspace(args.inc,args.inc*args.howlong,args.howlong)
plt.xlabel('Time')
plt.ylabel('l1 estimation errors')
plt.title('l1 estimation errors')
plt.plot(timeline,errors, label='PopART')
plt.fill_between(timeline, errors-vari_our, errors+vari_our, color='cornflowerblue')
plt.plot(timeline,errors_hao, label='Cmin-LASSO')
plt.fill_between(timeline, errors_hao-vari_hao, errors_hao+vari_hao, color='bisque')
plt.legend()
plt.show()