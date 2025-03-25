import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from numpy import linalg
from polynomial_regression import labels, examples
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = np.array([[0],[1],[2],[4]])
y = np.array([4,1,2,5])
poly = PolynomialFeatures(3)
fi = poly.fit_transform(X).astype(int)
fi_t=fi.transpose()
I=np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
product=np.matmul(fi_t,fi)
w=[]
lamb=[0,1,10]
for i in lamb:
    inv=linalg.inv(product+i*I)
    w.append(np.matmul(np.matmul(inv,fi_t),y))
for i in range(len(lamb)):
    print('Težine modela s faktorom ',lamb[i],': ',w[i])

w_ridge=[]
poly=PolynomialFeatures(3)
fi = poly.fit_transform(X)
from sklearn.linear_model import Ridge
for i in lamb:
    model=Ridge(alpha=i)
    model.fit(fi,y)
    coef=list(model.coef_)
    coef.pop(0)
    coef.insert(0,model.intercept_)
    w_ridge.append(coef)
for i in range(len(lamb)):
    print('Težine za model faktora ',lamb[i],': ',w_ridge[i])

lamb=[0,100]
d=[2,10]
colors=['blue','orange','yellow','green']
index=0
for i in lamb:
    for j in d:
        poly = PolynomialFeatures(j)
        fi = poly.fit_transform(examples)
        model = Ridge(alpha=i)
        model.fit(fi, labels)
        coef = list((model.coef_)[0])
        coef.pop(0)
        coef.insert(0, model.intercept_[0])
        pred=model.predict(fi)
        plt.plot(examples,pred,color=colors[index],label='h'+str(j)+' s faktorom '+str(i))
        index+=1
plt.scatter(examples,labels,color='red',label='(x,y)')
plt.xlabel('x')
plt.legend()
plt.show()

x_tr,x_te,y_tr,y_te=train_test_split(examples,labels,test_size=0.5)
err_test=[]
err_train=[]
lamb=range(51)
poly = PolynomialFeatures(10)
fi_train = poly.fit_transform(x_tr)
err_train=[]
err_test=[]
for i in lamb:
    model = Ridge(alpha=i)
    model.fit(fi_train, y_tr)
    pred = model.predict(fi_train)
    err_train.append(np.log(mean_squared_error(y_tr, pred)))
    fi_test = poly.fit_transform(x_te)
    pred = model.predict(fi_test)
    err_test.append(np.log(mean_squared_error(y_te, pred)))
plt.plot(lamb,err_train,color='red',label='ovisnost greške učenja o faktoru')
plt.plot(lamb,err_test,color='blue',label='ovisnost testne greške o faktoru')
plt.xlabel('Faktor regularizacije')
plt.legend()
plt.show()