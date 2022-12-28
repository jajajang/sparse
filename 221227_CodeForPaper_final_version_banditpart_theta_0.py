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
parser.add_argument("--d", type=int, help="dimension of the action vector", default=30)
parser.add_argument("--s", type=int, help="sparsity of the hidden parameter vector", default=2)
parser.add_argument("--same", type=bool, help='Whether we should use equal strategy which is ours', default=False)
parser.add_argument("--repeat",type=int, help='How many times it will repeat for each experiment setting', default=10)
parser.add_argument("--T0",type=int, help='Length of experiment to examine', default=10000)
parser.add_argument("--sigma",type=float, help='Variance', default=0.1)
parser.add_argument("--delta",type=float, help='Error probability in bandit', default=0.05)
parser.add_argument("--action_set", help='which action will I use - \'hard\' is Case 1 in the figure, and \'uniform\' is the Case 2 in the figure', default='hard')
parser.add_argument("--num_of_action_set", type=int, help='Number of actions in the action set in Case 2. Case 1 has fixed number of actions', default=90),
parser.add_argument("--cheat", type=float, help='theta_0 = cheat * theta', default=0)
args = parser.parse_args()


# functions for the optimization of lasso. Basically cvxpy format functions.

def loss_fn(X, Y, beta,n0):
    return cp.norm2(X @ beta - Y) ** 2 / n0

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd,n0):
    return loss_fn(X, Y, beta,n0) + lambd * regularizer(beta)

#basic settings
d=args.d
s=args.s
delta=args.delta

N_a = args.num_of_action_set
A= np.zeros((N_a,d))
# Case 1 action set: action set that satisfies M2 << Cmin^-1
# create action set that satisfies M2 << Cmin^-1
if args.action_set=='hard':
    N_a = d
    opt = 1 / np.sqrt(d)
    A=np.zeros((d,d))
    A[0][0]=opt
    for i in range(1,d):
        A[i][0]=1
        A[i][i]=opt

# Case 2 action set: Drawn uniformly from the unit sphere.
elif args.action_set=='uniform':
    A=np.zeros((N_a,d))
    for i in range(0,N_a):
        A[i]=np.random.normal(0,1,d)
        A[i]=A[i]/LA.norm(A[i])

else:
    print('Error: Invalid action set')


########################## Exploration plan optimization stage ################################

#Set1: Our Settings - Minimizing maximum diagonal entry of the inverse covariance matrix: max_i (Q^-1)_ii
mu=cp.Variable(N_a, pos=True)
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



#Set2: Setting of Botao Hao (2021) - Maximizing minimum eigenvalue of the covariance matrix
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


# To compare the pure estimation performance between Lasso and PopArt.
# Green line of the experiment, H2-Lasso is the setting when args.same=True

if args.same:
    p_first=mu_dist
    p_second=mu_dist
else:
    p_first=mu_dist
    p_second=dist_hao





#Experiment setup - variables
T0=args.T0                                      # Total number of rounds in bandit problem
repeative=args.repeat                           # How many times we repeat the same setting
sigma=args.sigma                                # Variance of the noise


hist_true=np.zeros((repeative,d))               #history of true theta
hist_esti=np.zeros((repeative,d))               #history of the estimated theta by our method
hist_esti_raw=np.zeros((repeative,d))           #history of the estimated theta by our method before thresholding
cum_regret=np.zeros((repeative,T0))             #Cumulative regret of our setting

hist_esti_hao=np.zeros((repeative,d))           #history of the estimated theta by Hao's method
cum_regret_hao=np.zeros((repeative,T0))         #Cumulative regret of the Hao's setting



for rep in range(0,repeative):
    #setting theta - changes over each experiment
    theta=np.zeros(d)
    if args.action_set=='uniform':
        indy=np.random.choice(d,s,replace=False)
        for indi in range(0,s):
            theta[indy[indi]]=1
    elif args.action_set=='hard':
        theta[0]=1
        theta[np.random.choice(d-1)+1]=1

    theta_0 = args.cheat*theta                  #PopArt can exploit pilot estimator when the agent has one. For basic setup, this cheat is always 0 and we didn't actually use it.
    a_true=A[np.argmax(A@theta)]

    # Experiment 1: PopArt experiment
    S=np.max(np.abs(A@(theta)))
    S_0=np.max(np.abs(A@(theta-theta_0)))
    vari = (S_0**2 + sigma**2)*M2
    T_exp = int(((s*T0/S)**2*vari*np.log(2*d/delta))**(1/3))

    threshold = width_catoni(T_exp,d,delta,vari)
    hist_true[rep]=theta
    X_hist=np.zeros((T_exp,d))
    cum_reg=0                                       #temporary variable for cumulative regret
    for t in range(0,T_exp):
        act_t=A[np.random.choice(N_a, p=p_first)]
        r=theta@act_t+np.random.normal(0,sigma)
        cum_reg+=theta@(a_true-act_t)
        cum_regret[rep][t]=cum_reg
        X_hist[t]=(r-act_t@theta_0)*(Q_inv@act_t)+theta_0
    theta_hat_raw=catoni_esti(X_hist,delta,vari)
    hist_esti_raw[rep]=theta_hat_raw
    theta_hat=(np.abs(theta_hat_raw)>threshold)*theta_hat_raw
    hist_esti[rep]=theta_hat
    for t in range(T_exp, T0):
        act_t=A[np.argmax(A@theta_hat)]
        cum_reg+=theta@(a_true-act_t)
        cum_regret[rep][t]=cum_reg


    #Experiment 2 - Botao Hao setting experiment
    T_exp_hao=int((2*(s*sigma*T0/S/Cmin)**2*np.log(d))**(1/3))
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
    lambd_b = 4 * sigma*np.sqrt(np.log(d)/ T_exp_hao)
    lassosol = cp.Problem(cp.Minimize(objective_fn(hist_b, r_b, beta, lambd_b,T_exp_hao)))
    lassosol.solve()
    beta_hat = beta.value
    hist_esti_hao[rep]=beta_hat
    a_hat_h=A[np.argmax(A@beta_hat)]
    for t in range(T_exp_hao,T0):
        act_h_t = A[np.argmax(A @ beta_hat)]
        cum_reg_hao += theta @ (a_true - act_h_t)
        cum_regret_hao[rep][t] = cum_reg_hao



# Plot Session

import matplotlib.pyplot as plt
timeline=list(range(T0))
plt.xlabel('Time Horizon')
plt.ylabel('Cumulative regret')
plt.title('Unit vec case, H_*=%.2f, 1/Cmin=%.2f'%(np.sqrt(M2), 1/Cmin))

vari_our=np.std(cum_regret,0)
mean_our=np.mean(cum_regret,0)
plt.plot(timeline,mean_our, label='PopART')
plt.fill_between(timeline, mean_our-vari_our, mean_our+vari_our, color='cornflowerblue',alpha=0.5)

vari_hao=np.std(cum_regret_hao,0)
mean_hao=np.mean(cum_regret_hao,0)
if args.same:
    plt.plot(timeline,mean_hao, label='H2-LASSO')
else:
    plt.plot(timeline,mean_hao, label='Cmin-LASSO')
plt.fill_between(timeline, mean_hao-vari_hao, mean_hao+vari_hao, color='bisque', alpha=0.5)
plt.legend()
plt.show()