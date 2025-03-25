import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.preprocessing import PolynomialFeatures


def nonzeroes(coef, tol=1e-6):
    return len(coef) - len(coef[np.isclose(0, coef, atol=tol)])

from sklearn.linear_model import Ridge
lamb=range(101)
poly = PolynomialFeatures(5)
fi_train = poly.fit_transform(x_train)
w_0=[]
w_1=[]
w_2=[]
for i in lamb:
    model = Ridge(alpha=i)
    model.fit(fi_train, y_train)
    coef = list(model.coef_[0])
    coef.pop(0)
    w_0.append(nonzeroes(np.array(coef)))
    l1=0
    l2=0
    for i in coef:
        l1+=abs(i)
        l2+=i**2
    w_1.append(l1)
    w_2.append(l2**0.5)
plt.plot(lamb,w_0,color='blue',label='ovisnost L-0 regularizacije o faktoru')
plt.plot(lamb,w_1,color='red',label='ovisnost L-1 regularizacije o faktoru')
plt.plot(lamb,w_2,color='green',label='ovisnost L-2 regularizacije o faktoru')
plt.xlabel('Faktor regularizacije')
plt.legend()
plt.show()