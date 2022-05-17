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
parser.add_argument("--same", type=bool, help='Whether we should use equal strategy which is ours', default=False)
parser.add_argument("--inc",type=int, help='Incremental scale of the experiment', default=1000)
parser.add_argument("--repeat",type=int, help='How many times it will repeat for each experiment setting', default=10)
parser.add_argument("--howlong",type=int, help='Maximum multiple of T_min for the experiment. For example, howlong=20 and inc=2000 means maximum 40000 rounds', default=10)
parser.add_argument("--T_min",type=int, help='Minimum length of experiment to examine', default=0)
parser.add_argument("--sigma",type=float, help='Variance', default=0.1)
parser.add_argument("--delta",type=float, help='Error probability in bandit', default=0.05)
parser.add_argument("--action_set", help='which action will I use', default='hard')
parser.add_argument("--cheat", type=float, help='theta_0 = cheat * theta', default=0)
parser.add_argument("--rho", type=float, help='rho^2 in that setting, 0.3 or 0.7', default=0.3)
args = parser.parse_args()


def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y) ** 2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


d=100
s=5
delta=args.delta
rho=args.rho
N_a = 20
#create action set that satisfies M2 << Cmin^-1
indy=np.random.choice(d,5,replace=False)
theta=np.zeros(d)
for i in range(s):
    theta[indy[i]]=np.random.uniform(0,1)


Q=(1-rho)*np.eye(d)+rho*np.reshape(np.ones(d*d),(d,d))
Cmin=np.min(LA.eig(Q)[0])
Q_inv=LA.inv(Q)
M2=np.max(np.diag(Q_inv))



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
        theta_0 = args.cheat*theta
        #same setting from here - threshold and catoni
        S=1
        S_0=1
        vari = (S_0**2 + sigma**2)*M2

        threshold = width_catoni(T0,d,delta,vari)
        hist_true[i*repeative+rep]=theta
        X_hist=np.zeros((T0,d))        
        hist_b=np.zeros((T0, d))                        #temporary history for the action of Hao's method, since it computes LASSO optimization
        r_b=np.zeros(T0)                                #temporary history for the reward

        for t in range(0,T0):
            A=np.random.multivariate_normal(np.zeros(d),Q,N_a)
            a_true=A[np.argmax(A@theta)]
            act_t=A[np.random.choice(N_a)]
            act_h_t=A[np.random.choice(N_a)]
            r=theta@act_t+np.random.normal(0,sigma)            
            X_hist[t]=(r-act_t@theta_0)*(Q_inv@act_t)+theta_0
            r_b[t] = theta @ act_h_t + np.random.normal(0, sigma)
            hist_b[t]=act_h_t
        theta_hat_raw=catoni_esti(X_hist,delta,vari)
        hist_esti_raw[i*repeative+rep]=theta_hat_raw
        theta_hat=(np.abs(theta_hat_raw)>threshold)*theta_hat_raw
        #theta_hat=LA.inv(X_hist)@theta_raw
        hist_esti[i*repeative+rep]=theta_hat
        all_error[i*repeative+rep]=LA.norm(theta-theta_hat,1)
        errors[i]+=LA.norm(theta-theta_hat,1)/repeative
        a_hat=A[np.argmax(A@theta_hat)]
        choice_errors[i]+=LA.norm(a_true-a_hat)/repeative

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
    vari_our[i]=np.std(all_error[repeative*i:repeative*(i+1)-1])
    vari_hao[i]=np.std(all_error_hao[repeative*i:repeative*(i+1)-1])

import matplotlib.pyplot as plt
timeline=np.linspace(args.inc,args.inc*args.howlong,args.howlong)
plt.xlabel('Time')
plt.ylabel('l1 estimation errors')
plt.title('Contextual, H_*=%.2f, 1/Cmin=%.2f, sigma=%.2f'%(np.sqrt(M2), 1/Cmin, sigma))
plt.plot(timeline,errors, label='PopART')
plt.errorbar(timeline, errors, vari_our, color='cornflowerblue', alpha=0.5)
np.savetxt("Contextual_H2-PopArt with d_%d_sigma_%.2f_H2=%.2f_Cmin_%.2f.csv"%(d, sigma, np.sqrt(M2), 1/Cmin), (errors, vari_our))
if args.same:
    plt.plot(timeline, errors_hao, label='H2-LASSO')
    np.savetxt("Contextual_H2-Lasso with d_%d_sigma_%.2f_H2=%.2f_Cmin_%.2f.csv"%(d, sigma, np.sqrt(M2), 1/Cmin), (errors_hao, vari_hao))
else:
    plt.plot(timeline,errors_hao, label='Cmin-LASSO')
    np.savetxt("Contextual_Cmin-Lasso with d_%d_sigma_%.2f_H2=%.2f_Cmin_%.2f.csv"%(d, sigma, np.sqrt(M2), 1/Cmin), (errors_hao, vari_hao))
plt.errorbar(timeline, errors_hao, vari_hao, color='bisque', alpha=0.5)

plt.legend()
plt.show()