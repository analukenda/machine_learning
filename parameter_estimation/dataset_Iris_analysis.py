import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import itertools as it

X,y=load_iris(return_X_y=True)
X=X[y==2]
y=y[y==2]
i=0
for (x_i,x_j) in it.combinations(X.T,2):
    plt.scatter(x_i,x_j)
    plt.show()

from scipy.stats import norm
def L_gauss(x, mi, sigma):
    # Vaš kôd ovdje...
    return np.log(np.prod(norm(loc=mi,scale=sigma).pdf(x)))

N=len(X[:,0])
for i in range(4):
    x_=X[:,i]
    mi=np.sum(x_)/N
    var=np.sum((x_-mi)**2)/N
    print('x'+str(i)+' mi='+str(mi)+', var='+str(var)+', log-izglednost='+str(L_gauss(x_,mi,(var)**0.5)))

from scipy.stats import pearsonr
for (x_1,x_2) in it.combinations(X.T,2):
    ro=pearsonr(x_1,x_2)
    print('ro='+str(ro[0]))

N=len(X)
Ns=[N//4,N//2,N]
abs_dif=[]
square_dif=[]
for N_i in Ns:
    x_=X[:N_i,]
    unbiased=np.diag(np.cov(x_.T))
    biased=[]
    for i in range(4):
        _x=x_[:,i]
        mu=np.sum(_x)/N_i
        biased.append(np.sum((_x-mu)**2)/N_i)
    abs_dif.append(np.abs(unbiased-biased))
    square_dif.append((unbiased-biased)**2)
abs_dif=np.array(abs_dif)
square_dif=np.array(square_dif)
fig,axs=plt.subplots(1,2,figsize=(15,6))
colors=['blue','orange','yellow','pink']
colors=['blue','orange','yellow','pink']
for i in range(4):
    axs[0].plot(Ns,abs_dif[:,i],color=colors[i])
    axs[1].plot(Ns,square_dif[:,i],color=colors[i])

legend_=['x'+str(i) for i in range(4)]
axs[0].legend(legend_)
axs[1].legend(legend_)
plt.show()