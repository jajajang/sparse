import scipy
import numpy as np

def psi(y):
    absy = np.abs(y)
    return np.sign(y) * np.log(1 + absy + absy ** 2 / 2)

def grad_catoni(X, y, alpha):
    return -psi(alpha * (X - y)).sum()

def catoni_findbracket(X,alpha):
    z0 = X.mean()
    g0 = grad_catoni(X, z0, alpha)
    if (g0 >= 0):
        ub = z0
        # now, find a lower bound
        i = 1
        while True:
            myz =  ub - 2**i
            g = grad_catoni(X, myz, alpha)
            if (g <= 0):
                break
        lb = myz
    else:
        lb = z0
        i = 1
        while True:
            myz =  lb + 2**i
            g = grad_catoni(X, myz, alpha)
            if (g >= 0):
                break
        ub = myz

    return [lb,ub]




def catoni_esti(X,delta,vari):
    T0=np.shape(X)[0]
    d=np.shape(X)[1]
    iota= np.log(2*d/delta)
    alpha = np.sqrt(2*iota/(vari*T0*(1+(2*iota)/(T0-2*iota))))

    esti=np.zeros(d)
    for i in range(0,d):
        lb,ub = catoni_findbracket(X[:,i],alpha)
        ret = scipy.optimize.root_scalar(lambda y: grad_catoni(X[:,i], y, alpha), x0=X[:,i].mean(),bracket=[lb,ub])
        esti[i]=ret.root
    return esti


def width_catoni(T0,d,delta,vari):
    iota = np.log(2 * d / delta)
    return np.sqrt(vari * 2 * iota / (T0 - 2*iota))

def support_check(pred,true):
    pred_labels = (np.abs(pred) > 0)
    true_labels = (np.abs(true) > 0)
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    return np.array([TP, TN, FP, FN])