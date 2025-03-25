from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from k_means import X1, X2, X3

K=[2,2,3]
x=[X1,X2,X3]
for i in range(len(K)):
    y=KMeans(n_clusters=K[i]).fit(x[i]).predict(x[i])
    plt.scatter(x[i][:,0],x[i][:,1],c=y,cmap=plt.get_cmap("cool"),s=20)
    plt.show()