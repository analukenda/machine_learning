import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from plot_silhouette import plot_silhouette

Xp, yp = make_blobs(n_samples=300, n_features=2, centers=[[0, 0], [3, 2.5], [0, 4]],
                    cluster_std=[0.45, 0.3, 0.45], random_state=96)
plt.scatter(Xp[:,0], Xp[:,1], c=yp, cmap=plt.get_cmap("cool"), s=20)

from sklearn.cluster import KMeans
K=range(1,16)
J=[]
for i in K:
    J.append(KMeans(n_clusters=i).fit(Xp).inertia_)
plt.plot(K,J)
plt.title("Ovisnost J o K")
plt.show()

import warnings
warnings.filterwarnings('ignore')
K=[2,3,5]
for k_i in K:
    plot_silhouette(k_i,Xp)
    plt.show()

from sklearn.datasets import make_blobs

X1, y1 = make_blobs(n_samples=1000, n_features=2, centers=[[0, 0], [1.3, 1.3]], cluster_std=[0.15, 0.5], random_state=96)
plt.scatter(X1[:,0], X1[:,1], c=y1, cmap=plt.get_cmap("cool"), s=20)

y=KMeans(n_clusters=2).fit(X1).predict(X1)
plt.scatter(X1[:,0],X1[:,1],c=y,cmap=plt.get_cmap("cool"),s=20)
plt.show()

from sklearn.datasets import make_circles

X2, y2 = make_circles(n_samples=1000, noise=0.15, factor=0.05, random_state=96)
plt.scatter(X2[:,0], X2[:,1], c=y2, cmap=plt.get_cmap("cool"), s=20)

y=KMeans(n_clusters=2).fit(X2).predict(X2)
plt.scatter(X2[:,0],X2[:,1],c=y,cmap=plt.get_cmap("cool"),s=20)
plt.show()

X31, y31 = make_blobs(n_samples=1000, n_features=2, centers=[[0, 0]], cluster_std=[0.2], random_state=69)
X32, y32 = make_blobs(n_samples=50, n_features=2, centers=[[0.7, 0.5]], cluster_std=[0.15], random_state=69)
X33, y33 = make_blobs(n_samples=600, n_features=2, centers=[[0.8, -0.4]], cluster_std=[0.2], random_state=69)
plt.scatter(X31[:,0], X31[:,1], c="#00FFFF", s=20)
plt.scatter(X32[:,0], X32[:,1], c="#F400F4", s=20)
plt.scatter(X33[:,0], X33[:,1], c="#8975FF", s=20)

# Just join all the groups in a single X.
X3 = np.vstack([X31, X32, X33])
y3 = np.hstack([y31, y32, y33])

y=KMeans(n_clusters=3).fit(X3).predict(X3)
plt.scatter(X3[:,0],X3[:,1],c=y,cmap=plt.get_cmap("cool"),s=20)
plt.show()
