import sklearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from polynomial_regression import make_instances, make_labels, func, labels, examples
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    err=[]
    d=[1,3,5,10,20]
    for i in zip(d,['grey','blue','yellow','orange','green']):
        poly = PolynomialFeatures(i[0])
        x_poly = poly.fit_transform(examples)
        model = LinearRegression()
        model.fit(x_poly, labels)
        pred = model.predict(x_poly)
        plt.plot(examples, pred, color=i[1],label='h'+str(i[0])+'(x)' )
        err.append(mean_squared_error(labels,pred))
    plt.scatter(examples, labels, color='red',label='(x,y)')
    plt.xlabel('x')
    plt.legend()
    plt.show()
    for i in range(len(d)):
        print('Greska modela stupnja ',d[i],': ',err[i])

    x_train, x_test, y_train, y_test = train_test_split(examples, labels, test_size=0.5)
    err_test=[]
    err_train=[]
    for i in range(1,21):
        poly = PolynomialFeatures(i)
        fi_train = poly.fit_transform(x_train)
        model = LinearRegression()
        model.fit(fi_train, y_train)
        pred=model.predict(fi_train)
        err_train.append(np.log(mean_squared_error(y_train,pred)))
        fi_test=poly.fit_transform(x_test)
        pred = model.predict(fi_test)
        err_test.append(np.log(mean_squared_error(y_test,pred)))
    plt.scatter(range(1,21),err_train,color='red',label='greška treniranja')
    plt.scatter(range(1,21),err_test,color='blue',label='greška testiranja')
    plt.legend()
    plt.show()

    svi_pr=make_instances(-5,5,1000)
    x_tr,x_te=train_test_split(svi_pr,test_size=0.5)
    sum=[100,200,500]
    y_tr=[]
    y_te=[]
    for i in sum:
        y_tr.append(make_labels(x_tr,func,i))
        y_te.append(make_labels(x_te, func, i))
    N=[166,333,500]
    indexi=range(500)
    fig, axs = plt.subplots(3,3,figsize=(20,20))
    for i in range(len(sum)):
        for j in range(len(N)):
            if N[j] != 500:
                rand_indexi = np.random.choice(indexi, N[j], replace=False)
                x_tr_rand=[]
                x_te_rand=[]
                y_tr_rand=[]
                y_te_rand=[]
                for k in rand_indexi:
                    x_tr_rand.append(x_tr[k])
                    x_te_rand.append(x_te[k])
                    y_tr_rand.append(y_tr[i][k])
                    y_te_rand.append(y_te[i][k])
            else:
                x_tr_rand = x_tr
                x_te_rand = x_te
                y_tr_rand = y_tr[i]
                y_te_rand = y_te[i]
            err_test = []
            err_train = []
            for k in range(1, 21):
                poly = PolynomialFeatures(k)
                fi_train = poly.fit_transform(x_tr_rand)
                model = LinearRegression()
                model.fit(fi_train, y_tr_rand)
                pred = model.predict(fi_train)
                err_train.append(np.log(mean_squared_error(y_tr_rand, pred)))
                fi_test = poly.fit_transform(x_te_rand)
                pred = model.predict(fi_test)
                err_test.append(np.log(mean_squared_error(y_te_rand, pred)))
            axs[i][j].plot(range(1, 21), err_train, color='red', label='greška treniranja')
            axs[i][j].plot(range(1, 21), err_test, color='blue', label='greška testiranja')
            axs[i][j].set_title('N = ' + str(N[j]) + ',šum = ' + str(sum[i]))
            axs[i][j].legend()
            plt.show()
