
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import normal
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def make_labels(X, f, noise=0):
    # Vaš kôd ovdje
    labels=[]
    for i in X:
        labels.append(f(i)+np.random.normal(0,noise))
    return labels

def make_instances(x1, x2, N) :
    return np.array([np.array([x]) for x in np.linspace(x1,x2,N)])

examples=make_instances(-5,5,50)
def func(x):
    return 5+x-2*x**2-5*x**3
labels=make_labels(examples,func,200)

if __name__=='__main__':
    plt.scatter(examples,labels,color='red')
    plt.xlabel('x')
    plt.ylabel('y')

    poly=PolynomialFeatures(3)
    x_poly=poly.fit_transform(examples)
    model=LinearRegression()
    model.fit(x_poly,labels)
    plt.scatter(examples, labels, color='red',label='(x,y)')
    plt.show()
    pred=model.predict(x_poly)
    plt.plot(examples, pred, color='blue',label='h(x)')
    plt.xlabel('x')
    plt.legend()
    plt.show()
    err=mean_squared_error(labels,pred)
    print('Greska modela ',err)









