#%%

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
import pandas as pd
import time


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
####FPTAS#####
# subfunction computeScore for computing f_i(x_i)
def computeScore(c, mu, beta, x, eta):
    # F= np.maximum()
    f = np.ceil(np.minimum(1, (beta * mu) / (c + x)) / eta) * eta
    return f



# subfunction invertScore for computing f_i(x_i)
def invertScore(c, mu, beta, f):
    x = np.maximum(0, (beta * mu) / f - c)
    print(x)
    return x


def minSumFPTASNoWealth(c, mu, beta, B, delta, epsilon):
    n = c.shape[0]

    C, _, psi0, psiOpt = minMaxBinarySearchNoWealth(c, mu, beta, B, delta / (2 * n), 0, 1)

    if C <= (delta / n):
        p = np.mean(psiOpt)

    else:
        eta = epsilon * C / (2 * n)
        K = int(np.ceil((n ** 3) / epsilon))
        keta = np.arange(K) * eta

        G = np.zeros((n, K))

        G[1, :] = invertScore(c[0], mu[0], beta[0], keta)
        print('')

        for j in range(1, n):

            time_1 = time.time()

            GprevFlip = np.fliplr(G[j - 1, :])
            subsAgentJ = invertScore(c[j], mu[j], beta[j], keta)
            for i in range(K):
                G[j, i] = np.min(GprevFlip[(K - i + 1):K + 1] + subsAgentJ[0:i + 1])

            t = time.time() - time_1

            print('Agents 1-%g: %g seconds\n' % (j, t))

        print('')

        bestk = np.where(G[n, :] < B)[0] - 1
        x = np.concatenate(([G[1, bestk + 1]], np.diff(G[:, bestk + 1])))
        psiOpt = psi(c, mu, beta, x)
        p = np.mean(psiOpt)

    return p, x, psi0, psiOpt, G

if __name__ == "__main__":
    N = 10
    delta = 0.01
    mu = np.maximum(1.0, 2000 + 100 * np.random.rand(N))
    beta = np.random.exponential(scale=0.5, size=(N,))
    c = np.maximum(1.0, 2000 + 500 * np.random.rand(N))
    #     B = np.linspace(10**3,10**5,101)
    B = 10000
    epsilon = 0.05

    p, x, psi0, psiOpt = minSumFPTASNoWealth(c, mu, beta, B, delta, epsilon)