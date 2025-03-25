import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from MLE import L

fig,axs=plt.subplots(3,3,figsize=(20,20))
alphas=[1,1.5,2]
betas=[1,1.5,2]
x=np.arange(0,1,0.001)

for i in range(len(alphas)):
    for j in range(len(betas)):
        axs[i][j].plot(x,beta.pdf(x,alphas[i],betas[j]))
        axs[i][j].set_title('alpha='+str(alphas[i])+', beta='+str(betas[j]))
plt.show()


def zdruzena(mu, N, m, alpha_, beta_):
    return L(mu, N, m) * beta.pdf(mu, alpha_, beta_)

N = 10
m = 9
fig, axs = plt.subplots(3, 3, figsize=(20, 20))
for i in range(len(alphas)):
    for j in range(len(betas)):
        axs[i][j].plot(x, [zdruzena(x_i, N, m, alphas[i], betas[j]) for x_i in x])
        axs[i][j].set_title('alpha=' + str(alphas[i]) + ', beta=' + str(betas[j]))
plt.show()

N=10
m=1
alpha_=2
beta_=2
plt.plot(x,[L(x_i,N,m) for x_i in x],color='blue',label='L')
plt.plot(x,beta.pdf(x,alpha_,beta_),color='red',label='beta')
plt.plot(x,[zdruzena(x_i,N,m,alpha_, beta_) for x_i in x],color='green',label='P(mu,D)')
plt.legend()
plt.show()
