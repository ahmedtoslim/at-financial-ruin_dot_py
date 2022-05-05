#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
import pandas as pd
import time
mydf = pd.read_csv('/Users/atoslim/Library/Containers/com.microsoft.Excel/Data/Desktop/filename.csv')

u = mydf['Initial Reseve (= 2 months Income - Assumed']

c = mydf['Income Range'] - mydf['Schck 5 - Pregnancy Related Emergency Cost (NYS Health data)']

np.random.seed(20)
beta = np.random.exponential(scale=0.1, size=431)
print(max(beta))
B = 0
delta = 0.00001
B_max =400000
legend_names = ['Actual Shocks', '10% Increase in Shocks', '20% Increase in Shocks']
r = (beta *u)/c
def ruinprob(beta, u,c):
    r = (beta * u) / c
    for i in range(len(r)):
        if r[i] > 1:
            r[i] = 1
        elif r[i] < 0:
            r[i] = 0
    return r
r = ruinprob(beta, u,c)
np.savetxt("prob.txt", r, delimiter="", newline='\n', fmt='%.22e')
# np.sum(r)
# def xx(r,30,20000):
# list to append to plot p_values vs people saved
p_values_list = []

# print(r)
mean = np.mean(r)
print(mean)
median = np.median(r)
print(median)
twentyfive = np.percentile(r, 25)
print(twentyfive)

# print (beta[1],u[1],c[1])
# defining functions:
def plot_p(interval_gap,B,B_max,legend_index):
    p_values = []
    b_values = []
    intervals = ((B_max-B)/interval_gap)+1
    B =B- interval_gap

    for i in range(intervals):
        B = B+interval_gap
        m = minMaxBinarySearch(c, u, beta,B , delta, 0, 1)
        p_values.append(m[0])
        b_values.append(B)
    p_values_list.append(p_values)
    plt.plot(b_values,p_values,label = legend_names[legend_index])

    # plt.show()
    for i in range(len(b_values)):
        if p_values[i] < 1:
            # print(b_values[i])
            # print(p_values[i])
            break
    return None

def computeSubsidies(c, u, beta, p):
    x = (((beta * u) / p) - c)
    for i  in range(len(x)):
        if x[i]<0:
            x[i]=0
    return x

def minMaxBinarySearch(c, u, beta, B, delta, pmin, pmax):
    #
    # for i in np.linspace(pmin, pmax,1):
    #     print(i)

    if (pmax - pmin) < delta:
        p = pmax
        x = computeSubsidies(c, u, beta, p)
        # print(x)
    else:

        p = float(pmin + pmax) / 2
        # print(p)
        x = computeSubsidies(c, u, beta, p)
        if np.sum(x) <= B:
            pmax = p
            #print(p)
        else:
            pmin = p
            #print(p)
        p,x = minMaxBinarySearch(c, u, beta, B, delta, pmin, pmax)
    m = [p, x]
    np.savetxt("subsidies.txt", x, delimiter="", newline='\n')
    return m



def knapSack(W, wt, val):
    n = len(val)
    start = time.time()
    table = [[0 for x in range(W + 1)] for x in range(n + 1)]

    for i in range(n + 1):
        for j in range(W + 1):
            if i == 0 or j == 0:
                table[i][j] = 0
            elif wt[i - 1] <= j:
                table[i][j] = max(val[i - 1]
                                  + table[i - 1][j - wt[i - 1]], table[i - 1][j])
            else:
                table[i][j] = table[i - 1][j]
    end = time.time()

    return table[n][W],end - start


if __name__ == "__main__":
    mylist = [1.0, 1.1, 1.2]
    c_list = []
    m_list = []
    r_list = []
    legend_index = 0
    for cbeta in mylist:
        c = mydf['Income Range'] - mydf['Schck 5 - Pregnancy Related Emergency Cost (NYS Health data)'] * cbeta
        m = minMaxBinarySearch(c, u, beta, B, delta, 0, 1)
        r = ruinprob(beta, u, c)
        c_list.append(c)
        m_list.append(m)
        r_list.append(r)

        # print(type(m[0]))
        # print("x: ",m[1])
        # print("sum of x: ",m[1].sum())
        # print("p: ",m[0])
        interval_gap = 20000
        plot_p(interval_gap, B, B_max,legend_index)
        legend_index+=1

    plt.xlabel('Budget')
    plt.ylabel('Prob. of ruin')
    plt.title('Prob. of Ruin vs Budget', fontsize=20)
    plt.legend()
    plt.show()
    legend_index = 0
    for cbeta in mylist:
        c = mydf['Income Range'] - mydf['Schck 5 - Pregnancy Related Emergency Cost (NYS Health data)'] * cbeta
        m = minMaxBinarySearch(c, u, beta, B, delta, 0, 1)
        r = ruinprob(beta, u, c)
        plt.scatter(m[1], r,label = legend_names[legend_index])
        legend_index+=1

    plt.xlabel('Subsidies')
    plt.ylabel('Individual Ruin Probabilities')
    plt.title('Individual Ruin Probabilities vs Subsidies', fontsize=20)
    plt.legend()
    plt.show()
    legend_index = 0
    for m in m_list:
        plt.plot(m[1],label = legend_names[legend_index])
        legend_index+=1
    plt.legend()
    plt.show()

    budget_list = []
    time_taken_count_list = []
    legend_index = 0
    for m in m_list:
        W = 400000
        x = m[1]
        print(type(x))
        wt = x[:431].tolist()
        for i in range(len(wt)):
          wt[i] = int(math.ceil(wt[i]))
          print(wt)
          val = [1]*431

        print("number of people given subsidy: ",knapSack(W, wt, val))
        ppl_saved_count = []
        time_taken_count = []
        budget = []
        for w in range(0,400000,20000):

            saved_count,time_taken = knapSack(w, wt, val)
            ppl_saved_count.append(saved_count)
            time_taken_count.append(time_taken)
            budget.append(w)
        ppl_saved_count = np.array(ppl_saved_count)
        time_taken_count = np.array(time_taken_count)
        budget = np.array(budget)
        np.savetxt("people_saved.txt",ppl_saved_count,delimiter="", newline='\n')
        np.savetxt("time_taken_count.txt", time_taken_count, delimiter="", newline='\n')
        np.savetxt("budget.txt", budget, delimiter="", newline='\n')

        print(budget,ppl_saved_count,time_taken_count)
        plt.plot(budget,ppl_saved_count,label=legend_names[legend_index])
        legend_index+=1
        # appending budget and time taken value for below plot
        budget_list.append(budget)
        time_taken_count_list.append(time_taken_count)

    plt.xlabel('Budget')
    plt.ylabel('People Saved')
    plt.title('People Saved vs Budget', fontsize=20)
    plt.xscale("log")
    plt.legend()
    plt.show()
    legend_index = 0
    for i in range(len(budget_list)):

        budget = budget_list[i]
        time_taken_count = time_taken_count_list[i]
        plt.plot(budget,time_taken_count,label=legend_names[legend_index])
        legend_index += 1

    plt.xlabel('Time Taken')
    plt.ylabel('Budget')
    plt.title('Budget vs Time Taken', fontsize=20)
    plt.legend()
    plt.show()
    print(type(p_values_list[0]))
    print(type(saved_count))
    print(saved_count)
    print(p_values_list)
    plt.plot(p_values_list[0],ppl_saved_count)
    plt.xlabel('Probability of Ruin')
    plt.ylabel('People Saved')
    plt.title('People Saved vs Probability of Ruin', fontsize=20)
    plt.show()




