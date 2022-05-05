#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
import pandas as pd
import time
# mydf = pd.read_csv('/Users/atoslim/Library/Containers/com.microsoft.Excel/Data/Desktop/filename.csv')

# subfunction psi for computing ruin probabilities
def psi(c, mu, beta, x):
    p = np.minimum(1.0, beta * mu / (c + x))
    return p

# subfunction ruinScore for computing ruin scores
def ruinScore(c, mu, beta, w, x):
    psi = (beta * mu) / (c + x)
    f = (w * psi) / (c + x)
    f[psi > 1] = 0
    return f

# subfunction ruinScoreinv for inverting ruin scores
def ruinScoreinv(c, mu, beta, w, f):
    x = np.maximum(0, np.sqrt((w * beta * mu) / f) - c)
    return x

# subfunction parseInputs
def parseInputs(varargin):
    if len(varargin) != 5:
        raise ValueError('The number of args should be 5, but not %d.' % (len(varargin)))

    c = varargin[0]
    if (not isinstance(c, np.ndarray)) or np.any(c < 0):
        raise ValueError("c must be a np array of nonnegative numbers.")
    n = c.shape[0]

    mu = varargin[1]
    if (not isinstance(mu, np.ndarray)) or np.any(mu < 0) or not (mu.shape[0] == n):
        raise ValueError("mu must be a vector of nonnegative numbers")

    mu = np.reshape(mu, c.shape)

    beta = varargin[2]

    if (not isinstance(beta, np.ndarray)) or np.any(beta < 0) or (not (beta.shape[0] == n)):
        raise ValueError('beta must be a vector of nonnegative numbers')
    beta = np.reshape(beta, c.shape)

    w = varargin[3]
    if w.shape[0] == 0:
        w = np.ones(c.shape) / c.shape[0]

    if (not isinstance(w, np.ndarray)) or np.any(w < 0) or (not (w.shape[0] == n)):
        raise ValueError('w must be a vector of nonnegative numbers')
    w = np.reshape(w, c.shape)


    B = varargin[4]
    if (not isinstance(B, (float, int))) or B < 0:
        raise ValueError("B must be a nonnegative scalar.")

    return c, mu, beta, w, B

    ##priorityMINSUM


def priorityAlgorithmMinSum(*varargin):
    c, mu, beta, w, B = parseInputs(varargin)
    x = np.zeros(c.shape)
    psi0 = psi(c, mu, beta, x)
    phi0 = np.sum(w * psi0)
    f0 = ruinScore(c, mu, beta, w, x)
    ind = np.argsort(-f0)
    psiOpt = psi0
    fOpt = f0

    for j in range(c.shape[0] - 1):
        extraCost = ruinScoreinv(c[ind[0:j + 1]]+x[ind[0:j + 1]], mu[ind[0:j + 1]], beta[ind[0:j + 1]], w[ind[0:j + 1]], fOpt[ind[0:j + 1]])
        totalExtraCost = np.sum(extraCost)

        if totalExtraCost <= B:
            x[ind[0:j + 1]] = x[ind[0:j + 1]] + extraCost
            fOpt[ind[0:j + 1]] = ruinScore(c[ind[0:j + 1]], mu[ind[0:j + 1]], beta[ind[0:j + 1]], w[ind[0:j + 1]], x[ind[0:j + 1]])
            psiOpt[ind[0:j + 1]] = psi(c[ind[0:j + 1]], mu[ind[0:j + 1]], beta[ind[0:j + 1]], x[ind[0:j + 1]])
            phiOpt = np.sum(w * psiOpt)
            B = B - totalExtraCost

        else:
            alpha = B/totalExtraCost
            x[ind[0:j + 1]] = x[ind[0:j + 1]] + alpha*extraCost
            fOpt[ind[0:j + 1]] = ruinScore(c[ind[0:j + 1]],mu[ind[0:j + 1]], beta[ind[0:j + 1]],w[ind[0:j + 1]], x[ind[0:j + 1]])
            psiOpt[ind[0:j + 1]] = psi(c[ind[0:j + 1]],mu[ind[0:j + 1]],beta[ind[0:j + 1]],x[ind[0:j + 1]])
            phiOpt = np.sum(w*psiOpt)
            return x,psi0,phi0,f0,psiOpt,phiOpt,fOpt

    fLow = fOpt[ind[-1]] / 2

    while B > 0:
        extraCost = ruinScoreinv(c[ind] + x[ind], mu[ind], beta[ind], w[ind], fLow)
        totalExtraCost = np.sum(extraCost)
        if totalExtraCost <= B:
            x[ind] = x[ind] + extraCost
            B = B - totalExtraCost
            fLow = fLow / 2
        else:
            alpha = B / totalExtraCost
            x[ind] = x[ind] + alpha * extraCost
            B = 0
        fOpt[ind] = ruinScore(c[ind],mu[ind],beta[ind],w[ind],x[ind])
        psiOpt[ind] = psi(c[ind],mu[ind],beta[ind],x[ind])
        phiOpt = np.sum(w*psiOpt)
    return x, psi0, phi0, f0, psiOpt, phiOpt, fOpt

if __name__ == "__main__":
    N = 431
    c = np.maximum(1.0, 2000 + 500 * np.random.rand(N))
    mu = np.maximum(1.0, 2000 + 100 * np.random.rand(N))
    beta = np.random.exponential(scale=0.5, size=(N,))
    w = np.array([])
    delta = 0.00001
    B = np.linspace(10 ** 3, 10 ** 5, 431)
    psiOpt = np.zeros((c.shape[0], B.shape[0]))
    phiOpt = np.zeros(B.shape[0])
    for j in range(B.shape[0]):
        _, _, _, _, psiOpt[:, j], phiOpt[j], _ = priorityAlgorithmMinSum(c, mu, beta, np.array([]), B[j])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(B / c.shape[0], np.max(psiOpt, axis=0), color='red', linewidth=2, label='Maximum')
    ax.plot(B / c.shape[0], phiOpt, color='blue', linewidth=2, label='Average')

    ax.set_xlabel('Total Budget ($) per Pay Period / # Agents')
    ax.set_ylabel('Ruin Probability')
    ax.set_title('Prioritization Based on Min-Sum Objective')
    ax.grid(True)
    ax.set_ylim(0, 1)

    percentile_25 = np.array([np.percentile(psiOpt[:, j], 25) for j in range(psiOpt.shape[1])])
    percentile_50 = np.array([np.percentile(psiOpt[:, j], 50) for j in range(psiOpt.shape[1])])
    percentile_75 = np.array([np.percentile(psiOpt[:, j], 75) for j in range(psiOpt.shape[1])])

    ax.plot(B / c.shape[0], percentile_50, color='black', linewidth=1, linestyle='-', label='Median')
    ax.plot(B / c.shape[0], percentile_75, color='black', linewidth=1, linestyle='--', label='Upper/Lower Quartiles')
    ax.plot(B / c.shape[0], percentile_25, color='black', linewidth=1, linestyle='--')

    ax.legend();  # Add a legend.
    plt.show()