# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
import pandas as pd
import time


######MinMaxPriorityAlg######

# subfunction psi for computing ruin probabilities
def psi(c, mu, beta, x):
    p = np.minimum(1.0, beta * mu / (c + x))
    return p


# subfunction psiinv for inverting ruin probabilities
def psiinv(c, mu, beta, p):
    x = (beta * mu) / p - c
    return x


# subfunction parseInputs
def parseInputs(varargin):
    if len(varargin) != 4:
        raise ValueError('The number of args should be 4, but not %d.' % (len(varargin)))

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

    B = varargin[3]
    if (isinstance(B, int)) or B < 0:
        raise ValueError("B must be a nonnegative scalar.")

    return c, mu, beta, B


# priorityMINMAX

def priorityAlgorithmMinMax(*varargin):
    c, mu, beta, B = parseInputs(varargin)

    x = np.zeros(c.shape[0])
    psi0 = psi(c, mu, beta, x)

    ind = np.argsort(-psi0)

    psiOpt = psi0

    for j in range(c.shape[0] - 1):
        extraCost = psiinv(c[ind[0:j + 1]] + x[ind[0:j + 1]], mu[ind[0:j + 1]], beta[ind[0:j + 1]], psiOpt[ind[j + 1]])
        totalExtraCost = np.sum(extraCost)

        if totalExtraCost <= B:
            x[ind[0:j + 1]] = x[ind[0:j + 1]] + extraCost
            psiOpt[ind[0:j + 1]] = psi(c[ind[0:j + 1]], mu[ind[0:j + 1]], beta[ind[0:j + 1]], x[ind[0:j + 1]])

            B = B - totalExtraCost
        else:
            alpha = B / totalExtraCost
            x[ind[0:j + 1]] = x[ind[0:j + 1]] + alpha * extraCost
            psiOpt[ind[0:j + 1]] = psi(c[ind[0:j + 1]], mu[ind[0:j + 1]], beta[ind[0:j + 1]], x[ind[0:j + 1]])
            return x, psi0, psiOpt

    psiLow = psiOpt[ind[-1]] / 2
    while B > 0:
        extraCost = psiinv(c[ind] + x[ind], mu[ind], beta[ind], psiLow)
        totalExtraCost = np.sum(extraCost)
        if totalExtraCost <= B:

            x[ind] = x[ind] + extraCost
            B = B - totalExtraCost
            psiLow = psiLow / 2
        else:
            alpha = B / totalExtraCost
            x[ind] = x[ind] + alpha * extraCost
            B = 0
        psiOpt[ind] = psi(c[ind], mu[ind], beta[ind], x[ind])
    return x, psi0, psiOpt


if __name__ == "__main__":
    N = 10
    delta = 0.00001
    c = np.maximum(1.0, 2000 + 500 * np.random.rand(N))
    mu = np.maximum(1.0, 2000 + 100 * np.random.rand(N))

    beta = np.random.exponential(scale=0.5, size=(N,))
    print(beta)
    for i in range(len(beta)):
        if beta[i] > 1:
            beta[i] = 1
    B = np.linspace(10 ** 3, 10 ** 5, 101)
    p = np.zeros(B.shape[0])
    psiOpt = np.zeros((c.shape[0], B.shape[0]))
    t = np.zeros(B.shape[0])
    for j in range(B.shape[0]):
        _, _, psiOpt[:, j] = priorityAlgorithmMinMax(c, mu, beta, B[j])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(B/c.shape[0], np.max(psiOpt, axis=0), color='red', linewidth=2, label = 'Maximum')
    ax.plot(B/c.shape[0], np.mean(psiOpt, axis=0), color='blue', linewidth=2, label = 'Average')

    ax.set_xlabel('Total Budget ($) per Pay Period / # Agents')
    ax.set_ylabel('Ruin Probability')
    ax.set_title('Prioritization Based on Min-Max Objective')
    ax.grid(True)
    ax.set_ylim(0, 1)

    percentile_25 = np.array( [np.percentile(psiOpt[:,j], 25) for j in range(psiOpt.shape[1])] )
    percentile_50 = np.array( [np.percentile(psiOpt[:,j], 50) for j in range(psiOpt.shape[1])] )
    percentile_75 = np.array( [np.percentile(psiOpt[:,j], 75) for j in range(psiOpt.shape[1])] )

    ax.plot(B/c.shape[0], percentile_50, color = 'black', linewidth = 1, linestyle='-', label = 'Median')
    ax.plot(B/c.shape[0], percentile_75, color = 'black', linewidth = 1, linestyle='--', label = 'Upper/Lower Quartiles')
    ax.plot(B/c.shape[0], percentile_25, color = 'black', linewidth = 1, linestyle='--')

    ax.legend()  # Add a legend.
    plt.show()