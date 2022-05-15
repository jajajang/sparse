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
parser.add_argument("--repeat",type=int, help='How many times it will repeat for each experiment setting', default=30)
parser.add_argument("--T0",type=int, help='Length of experiment to examine', default=10000)
parser.add_argument("--sigma",type=float, help='Variance', default=1)
parser.add_argument("--delta",type=float, help='Error probability in bandit', default=0.05)
parser.add_argument("--multiplier", type=float, help='Scale exploration time', default=1)
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
N_a=d
opt=1/np.sqrt(d)
#create action set that satisfies M2 << Cmin^-1


A=np.zeros((N_a,d))
'''
for i in range(0,N_a):
    A[i]=np.random.normal(0,1,d)
    A[i]=A[i]/LA.norm(A[i])
'''
A[0][0]=opt
for i in range(1,d):
    A[i][0]=1
    A[i][i]=opt

#Set1: Our Settings
'''
mu=cp.Variable(d, pos=True)
X=A.T@ cp.diag(mu) @A

T = Variable((d, d), symmetric=True)
M = bmat([[X, np.eye(d)],
          [np.eye(d), T]])
constraints = [M >> 0, cp.sum(mu)==1]
objective = cp.Minimize(cp.max(cp.diag(T)))

prob=cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK,verbose=True)
'''
#results of set1

mu=cp.Variable(N_a, pos=True)
X=A.T@ cp.diag(mu) @A

T = Variable((d, d), symmetric=True)
M = bmat([[X, np.eye(d)],
          [np.eye(d), T]])
constraints = [M >> 0, cp.sum(mu)==1]
objective = cp.Minimize(cp.max(cp.diag(T)))

prob=cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK,verbose=True)

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
'''
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
'''

mu_hao=cp.Variable(N_a, pos=True)
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
T0=args.T0
repeative=args.repeat
sigma=args.sigma


hist_true=np.zeros((repeative,d))             #history of true theta
hist_esti=np.zeros((repeative,d))             #history of the estimated theta by our method
hist_esti_raw=np.zeros((repeative,d))
cum_regret=np.zeros((repeative,T0))

hist_esti_hao=np.zeros((repeative,d))         #history of the estimated theta by Hao's method
cum_regret_hao=np.zeros((repeative,T0))

hist_act_oful=np.zeros((repeative,T0,d))
cum_regret_oful=np.zeros((repeative,T0))
for rep in range(0,repeative):
    #setting theta - changes over each experiment
    print('Try!')
    #theta=A[np.random.randint(d)]
    theta = np.zeros(d)
    theta[0]=1
    theta[np.random.choice(d-1)+1]=1

    #same setting from here - threshold and catoni
    a_true=A[np.argmax(A@theta)]
    S=np.max(np.abs(A@theta))
    vari = (S**2 + sigma**2)*M2
    T_exp = int(args.multiplier*((s*T0/S)**2*vari*np.log(2*d/delta))**(1/3))
    T_exp = np.min((T0, T_exp))

    threshold = width_catoni(T_exp,d,delta,vari)
    hist_true[rep]=theta
    X_hist=np.zeros((T_exp,d))
    cum_reg=0                                       #temporary variable for cumulative regret
    for t in range(0,T_exp):
        act_t=A[np.random.choice(N_a, p=p_first)]
        r=theta@act_t+np.random.normal(0,sigma)
        cum_reg+=theta@(a_true-act_t)
        cum_regret[rep][t]=cum_reg
        X_hist[t]=r*(Q_inv@act_t)
    theta_hat_raw=catoni_esti(X_hist,delta,vari)
    hist_esti_raw[rep]=theta_hat_raw
    theta_hat=(np.abs(theta_hat_raw)>threshold)*theta_hat_raw
    hist_esti[rep]=theta_hat
    for t in range(T_exp, T0):
        act_t=A[np.argmax(A@theta_hat)]
        cum_reg+=theta@(a_true-act_t)
        cum_regret[rep][t]=cum_reg


    T_exp_hao=int(args.multiplier*(2*(s*sigma*T0/S/Cmin)**2*np.log(d))**(1/3))
    T_exp_hao=np.min((T0, T_exp_hao))
    hist_b=np.zeros((T_exp_hao, d))                        #temporary history for the action of Hao's method, since it computes LASSO optimization
    r_b=np.zeros(T_exp_hao)                                #temporary history for the reward

    cum_reg_hao=0
    for t in range(0,T_exp_hao):
        act_h_t = A[np.random.choice(N_a, p=p_second)]
        hist_b[t]=act_h_t
        r_b[t] = theta @ act_h_t + np.random.normal(0, sigma)
        cum_reg_hao+=theta@(a_true-act_h_t)
        cum_regret_hao[rep][t]=cum_reg_hao
    beta = cp.Variable(d)
    lambd_b = 4 * sigma*np.sqrt(np.log(d) * T_exp_hao)
    lassosol = cp.Problem(cp.Minimize(objective_fn(hist_b, r_b, beta, lambd_b)))
    lassosol.solve()
    beta_hat = beta.value
    hist_esti_hao[rep]=beta_hat
    a_hat_h=A[np.argmax(A@beta_hat)]
    for t in range(T_exp_hao,T0):
        act_h_t = A[np.argmax(A @ beta_hat)]
        cum_reg_hao += theta @ (a_true - act_h_t)
        cum_regret_hao[rep][t] = cum_reg_hao


###########OFUL
    cum_reg_oful=0
    V=np.eye(d)
    V_inv=LA.inv(V)
    theta_oful=np.zeros(d)
    b_t=np.zeros(d)
    logdetV0=LA.det(V)
    logdetV=logdetV0
    for t in range(0,T0):
        betty= sigma * np.sqrt(logdetV - logdetV0 + np.log(1 / (delta ** 2))) +np.norm(theta)
        a_oful_t=A[np.argmax(A@theta_oful+betty*np.sqrt(np.diag(A@V_inv@A.T)))]
        hist_act_oful[rep][t]=a_oful_t
        r_t = a_oful_t@theta + np.random.normal(0,sigma)
        cum_reg_oful+=theta@(a_true-a_oful_t)
        cum_regret_oful[rep][t]=cum_reg_oful

        V+=np.outer(a_oful_t, a_oful_t)
        logdetV=LA.det(V)
        b_t+=r_t*a_oful_t
        V_inv=LA.inv(V)
        theta_oful=V_inv@b_t


import matplotlib.pyplot as plt
timeline=list(range(T0))
plt.xlabel('Time Horizon')
plt.ylabel('Cumulative regret')
plt.title('Bandit, H_*=%.2f, 1/Cmin=%.2f, sigma=%.2f'%(np.sqrt(M2), 1/Cmin, sigma))

vari_our=np.std(cum_regret,0)
mean_our=np.mean(cum_regret,0)
plt.plot(timeline,mean_our, label='PopART')
plt.fill_between(timeline, mean_our-vari_our, mean_our+vari_our, color='cornflowerblue', alpha=0.5)

vari_hao=np.std(cum_regret_hao,0)
mean_hao=np.mean(cum_regret_hao,0)
plt.plot(timeline,mean_hao, label='Cmin-LASSO')
plt.fill_between(timeline, mean_hao-vari_hao, mean_hao+vari_hao, color='bisque', alpha=0.5)
'''
vari_oful=np.std(cum_regret_oful,0)
mean_oful=np.mean(cum_regret_oful,0)
plt.plot(timeline, mean_oful, label='OFUL')
plt.fill_between(timeline, mean_oful-vari_oful, mean_oful+vari_oful, color='green')
'''
plt.legend()
plt.show()