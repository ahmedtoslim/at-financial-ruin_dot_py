#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
import pandas as pd
import time
mydf = pd.read_csv('/Users/atoslim/Library/Containers/com.microsoft.Excel/Data/Desktop/filename.csv')

#####BINARY SEARCH#####

# subfunction psi for computing ruin probabilities
def psi(c, mu, beta, x):
    p = np.minimum(1.0, beta * mu / (c + x))
    return p


# subfunction computeSubsidies for computing subsidies
def computeSubsidies(c, mu, beta, p):
    x = np.maximum(0.0, beta * mu / p - c)
    return x


# Binary Search
def minMaxBinarySearchNoWealth(c, mu, beta, B, delta, pmin, pmax):
    # c: (n,)
    # mu: (n,)
    # beta: (n,)
    # B, delta, pmin, pmax: double
    # x, psi0, psiOpt: (n, )

    if (pmax - pmin) < delta:
        p = pmax
        x = computeSubsidies(c, mu, beta, p)
        psi0 = psi(c, mu, beta, np.zeros(x.shape))
        psiOpt = psi(c, mu, beta, x)
    else:
        p = float(pmin + pmax) / 2
        x = computeSubsidies(c, mu, beta, p)
        if np.sum(x) <= B:
            pmax = p
        else:
            pmin = p

        p, x, psi0, psiOpt = minMaxBinarySearchNoWealth(c, mu, beta, B, delta, pmin, pmax)
    return p, x, psi0, psiOpt



if __name__ == "__main__":
    N = 431
    delta = 0.00001
    mu = np.maximum(1.0, 4000 + 100 * np.random.rand(N))
    beta = np.random.exponential(scale=0.5, size=(N,))
    c = mydf['Income Range'] - mydf['Schck 5 - Pregnancy Related Emergency Cost (NYS Health data)']

    # c = np.maximum(1.0, 2000 + 500 * np.random.rand(N))
    B = np.linspace(10**3,10**5,431)
    p = np.zeros(B.shape[0])
    psiOpt = np.zeros((c.shape[0], B.shape[0]))
    tb = np.zeros(B.shape[0])
    for j in range(B.shape[0]):
        p[j],_,_,psiOpt[:,j] = minMaxBinarySearchNoWealth(c,mu,beta,B[j],delta,0,1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(B/c.shape[0], np.max(psiOpt, axis=0), color='red', linewidth=2, label = 'Maximum')
    ax.plot(B/c.shape[0], np.mean(psiOpt, axis=0), color='blue', linewidth=2, label = 'Average')

    ax.set_xlabel('Total Budget ($) per Pay Period / # Agents')
    ax.set_ylabel('Ruin Probability')
    ax.set_title('Prioritization Based on Min-Max Objective via Binary Search')
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

