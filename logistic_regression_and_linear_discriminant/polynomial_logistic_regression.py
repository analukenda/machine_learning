import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from plot_2d_clf import plot_2d_clf_problem
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X,y=make_classification(n_samples=100,n_redundant=0,n_features=2,n_classes=2,n_clusters_per_class=2)
plot_2d_clf_problem(X,y)
plt.show()
def h(x,poly):
    y=[]
    x_p=poly.fit_transform(x)
    for xi in x_p:
        y.append(model.predict([xi]))
    return np.array(y)
for i in [2,3]:
    model=LogisticRegression()
    poly=PolynomialFeatures(i)
    x_poly=poly.fit_transform(X)
    model.fit(x_poly,y)
    plot_2d_clf_problem(X,y,lambda x:h(x,poly))
    plt.show()