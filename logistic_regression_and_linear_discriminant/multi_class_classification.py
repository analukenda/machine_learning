import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from plot_2d_clf import plot_2d_clf_problem
from sklearn.linear_model import LinearRegression, RidgeClassifier

X,y=make_classification(n_features=2,n_redundant=0,n_classes=3,n_clusters_per_class=1)
plot_2d_clf_problem(X,y)
plt.show()

h=[]
for i in range(3):
    y_ovr=[]
    for label in y:
        if label==i:
            y_ovr.append(1)
        else:
            y_ovr.append(0)
    h_lin = LinearRegression()
    h_lin.fit(X, y_ovr)
    granica = lambda x: h_lin.predict(x) >= 0.5
    h.append(h_lin)
    plot_2d_clf_problem(X, y, granica)
    plt.show()

def predict(x):
    y=[]
    for xi in x:
        hj=[]
        for bin_class in h:
            hj.append(bin_class.predict([xi]))
        y.append(hj.index(max(hj)))
    return np.array(y)
plot_2d_clf_problem(X, y, predict)
plt.show()

h=RidgeClassifier(alpha=0)
h.fit(X,y)
plot_2d_clf_problem(X,y,h.predict)
plt.show()