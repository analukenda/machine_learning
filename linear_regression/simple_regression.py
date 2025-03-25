import numpy as np
import sklearn
import matplotlib.pyplot as plt

X = np.array([[0],[1],[2],[4]])
y = np.array([4,1,2,5])

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(1)
fi = poly.fit_transform(X).astype(int)
print(fi)

from numpy import linalg
fi_t=fi.transpose()
w_a=np.matmul(np.matmul(linalg.inv(np.matmul(fi_t,fi)),fi_t),y)
pseduo_inv=linalg.pinv(fi)
w_b=np.matmul(pseduo_inv,y)
print('Tezine izracunate formulom ',w_a)
print('Tezine izracunate pseudoinverzom ',w_b)

h=w_a[0]+w_a[1]*X
plt.plot(X,h,color='blue',label='h(x)')
plt.scatter(X,y,color='red',label='(x,y)')
from sklearn.metrics import mean_squared_error
err=mean_squared_error(y,h)
print('Greska modela ',err)
plt.xlabel('x')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X,y)
print(model.intercept_,model.coef_[0])



