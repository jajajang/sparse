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

'''

'''


parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, help="dimension of the action vector", default=10)
parser.add_argument("--s", type=int, help="sparsity of the hidden parameter vector", default=2)
parser.add_argument("--same", type=bool, help='Whether we should use equal strategy which is ours', default=False)
parser.add_argument("--inc",type=int, help='Incremental scale of the experiment', default=1000)
parser.add_argument("--T_min",type=int, help='Minimum length of experiment to examine', default=0)
parser.add_argument("--howlong",type=int, help='Maximum multiple of T_min for the experiment. For example, howlong=20 and inc=2000 means maximum 40000 rounds', default=10)
parser.add_argument("--repeat",type=int, help='How many times it will repeat for each experiment setting', default=10)
parser.add_argument("--sigma",type=float, help='Variance', default=0.1)
parser.add_argument("--delta",type=float, help='Error probability in bandit', default=0.05)
parser.add_argument("--action_set", help='which action will I use - \'hard\' is Case 1 in the figure, and \'uniform\' is the Case 2 in the figure', default='hard')
parser.add_argument("--num_of_action_set", type=int, help='Number of actions in the action set in Case 2. Case 1 has fixed number of actions', default=150),
parser.add_argument("--cheat", type=float, help='theta_0 = cheat * theta', default=0)
args = parser.parse_args()


# functions for the optimization of lasso. Basically cvxpy format functions.
def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y) ** 2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)



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

T0=args.T_min                                   # Initial number of samples for regression
repeative=args.repeat                           # How many times we repeat the same setting
howlong=args.howlong                            # How many cases of 'number of samples' we want to observe.
                                                # For example, if T_min=1000 and howlong=3,
                                                # we check the cases of T=1000,2000,3000
wholelength=repeative*howlong                   # The maximum number of the experiments
sigma=args.sigma                                # Variance of the noise


hist_true=np.zeros((wholelength,d))             #history of true theta, true hidden parameters

hist_esti=np.zeros((wholelength,d))             #history of the estimated theta by our method
hist_esti_raw=np.zeros((wholelength,d))         #history of the estimated theta by our method before thresholding
errors=np.zeros(howlong)                        #l1 error between true and our estimation
all_error=np.zeros(wholelength)                 #history of the error of all experiments. Used it for measuring variance
vari_our=np.zeros(howlong)                      #history of the variance of the l1 norm difference

hist_esti_hao=np.zeros((wholelength,d))         #history of the estimated theta by Hao's method
errors_hao=np.zeros(howlong)                    #l1 error between true and Hao's estimation
all_error_hao=np.zeros(wholelength)             #history of the error of all experiments with Hao's setting. Used it for measuring variance
vari_hao=np.zeros(howlong)                      #history of the variance of the l1 norm difference in Hao's setting

for i in range(0,howlong):
    T0=T0+args.inc
    print('=====Repetition for total time %d=====' % (T0))
    for rep in range(0,repeative):
        #setting theta - changes over each experiment
        theta = np.zeros(d)
        theta[0]=-1
        theta[np.random.choice(d-1)+1]=1
        theta_0 = args.cheat*theta              #PopArt can exploit pilot estimator when the agent has one. For basic setup, this cheat is always 0 and we didn't actually use it.
        a_true=A[np.argmax(A@theta)]
        S=np.max(np.abs(A@theta))

        #Experiment 1: PopArt experiment
        S_0=np.max(np.abs(A@(theta-theta_0)))
        vari = (S_0**2 + sigma**2)*M2

        threshold = width_catoni(T0,d,delta,vari)
        hist_true[i*repeative+rep]=theta
        X_hist=np.zeros((T0,d))
        for t in range(0,T0):
            act_t=A[np.random.choice(N_a, p=p_first)]
            r=theta@act_t+np.random.normal(0,sigma)
            X_hist[t]=(r-act_t@theta_0)*(Q_inv@act_t)+theta_0
        theta_hat_raw=catoni_esti(X_hist,delta,vari)
        hist_esti_raw[i*repeative+rep]=theta_hat_raw
        theta_hat=(np.abs(theta_hat_raw)>threshold)*theta_hat_raw
        #theta_hat=LA.inv(X_hist)@theta_raw
        hist_esti[i*repeative+rep]=theta_hat
        all_error[i*repeative+rep]=LA.norm(theta-theta_hat,1)
        errors[i]+=LA.norm(theta-theta_hat,1)/repeative
        a_hat=A[np.argmax(A@theta_hat)]

        #Experiment 2 - Botao Hao setting experiment
        hist_b=np.zeros((T0, d))                        #temporary history for the action of Hao's method, since it computes LASSO optimization
        r_b=np.zeros(T0)                                #temporary history for the reward
        for t in range(0,T0):
            act_h_t = A[np.random.choice(N_a, p=p_second)]
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
    vari_our[i]=np.std(all_error[repeative*i:repeative*(i+1)-1])
    vari_hao[i]=np.std(all_error_hao[repeative*i:repeative*(i+1)-1])



# Plot session

import matplotlib.pyplot as plt
timeline=np.linspace(args.inc,args.inc*args.howlong,args.howlong)
plt.xlabel('Time')
plt.ylabel('l1 estimation errors')
plt.title('H_*=%.2f, 1/Cmin=%.2f, sigma=%.2f'%(np.sqrt(M2), 1/Cmin, sigma))
plt.plot(timeline,errors, label='PopART')
plt.errorbar(timeline, errors, vari_our, color='cornflowerblue', alpha=0.5)
np.savetxt("H2-PopArt with d_%d_sigma_%.2f_H2=%.2f_Cmin_%.2f.csv"%(d, sigma, np.sqrt(M2), 1/Cmin), (errors, vari_our))
if args.same:
    plt.plot(timeline, errors_hao, label='H2-LASSO')
    np.savetxt("H2-Lasso with d_%d_sigma_%.2f_H2=%.2f_Cmin_%.2f.csv"%(d, sigma, np.sqrt(M2), 1/Cmin), (errors_hao, vari_hao))
else:
    plt.plot(timeline,errors_hao, label='Cmin-LASSO')
    np.savetxt("Cmin-Lasso with d_%d_sigma_%.2f_H2=%.2f_Cmin_%.2f.csv"%(d, sigma, np.sqrt(M2), 1/Cmin), (errors_hao, vari_hao))
plt.errorbar(timeline, errors_hao, vari_hao, color='bisque', alpha=0.5)

plt.legend()
plt.show()