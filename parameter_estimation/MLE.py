import math

import numpy as np
from matplotlib import pyplot as plt


def L(mu, N, m):
    return math.pow(mu,m)*math.pow(1-mu,N-m)

if __name__=='__main--':
    Ns=[10,100]
    ms=[[1,2,5,9],[1,10,50,90]]
    number_ms=len(ms[0])
    x=np.arange(0,1,0.001)
    for i in range(2):
        colors=['blue','red','orange','green']
        for m in range(number_ms):
            plt.plot(x,[L(x_i,Ns[i],ms[i][m]) for x_i in x],color=colors[m])
        plt.legend(ms[i])
        plt.title('Ovisnost izglednosti o mu, N i m')
        plt.show()

    N=10
    colors=['blue','red']
    ms=[0,9]
    for i in range(len(ms)):
        plt.plot(x,[L(x_i,N,ms[i]) for x_i in x],color=colors[i])
    plt.legend(ms)
    plt.title('Ovisnost izglednosti o mu, N i m')
    plt.show()

