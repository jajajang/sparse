import scipy
import numpy as np

def psi(y):
    absy = np.abs(y)
    return np.sign(y) * np.log(1 + absy + absy ** 2 / 2)

def grad_cantoni(X, y, alpha):
    return -psi(alpha * (X - y)).sum()


def catoni_esti(X,delta,M2):
    T0=np.shape(X)[0]
    d=np.shape(X)[1]
    vari=np.sqrt(M2/T0)
    iota= np.log(2*d/delta)
    alpha = np.sqrt(2*iota/(vari*T0*(1+(2*iota)/(T0-2*iota))))

    esti=np.zeros(d)
    for i in range(0,d):
        ret = scipy.optimize.root_scalar(lambda y: grad_cantoni(X[:,i], y, alpha), x0=X[:,i].mean())
        esti[i]=ret.root
    return esti


def width_catoni(X,delta,M2):
    T0 = np.shape(X)[0]
    d = np.shape(X)[1]
    vari = np.sqrt(M2 / T0)
    iota = np.log(2 * d / delta)
    alpha = np.sqrt(2 * iota / (vari * T0 * (1 + (2 * iota) / (T0 - 2 * iota))))
    return np.sqrt(vari * 2 * iota / (T0 - iota))

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
    return [TP, TN, FP, FN]