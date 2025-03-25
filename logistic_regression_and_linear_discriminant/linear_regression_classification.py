from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from plot_2d_clf import plot_2d_clf_problem

seven_X = np.array([[2,1], [2,3], [1,2], [3,2], [5,2], [5,4], [6,3]])
seven_y = np.array([1, 1, 1, 1, 0, 0, 0])

unsep_X = np.append(seven_X, [[2,2]], axis=0)
unsep_y = np.append(seven_y, 0)

if __name__=='__main__':

    h=RidgeClassifier(alpha=0)
    h.fit(seven_X,seven_y)
    plot_2d_clf_problem(seven_X,seven_y,h.predict)
    plt.show()
    acc=accuracy_score(seven_y,h.predict(seven_X))
    print('Tocnost: '+str(acc))

    from sklearn.metrics import mean_squared_error
    h_lin=LinearRegression()
    h_lin.fit(seven_X,seven_y)
    granica=lambda x:h_lin.predict(x)>=0.5
    plot_2d_clf_problem(seven_X,seven_y,granica)
    plt.show()
    acc=1-mean_squared_error(seven_y,h_lin.predict(seven_X))
    print('Tocnost: '+str(acc))

    outlier_X = np.append(seven_X, [[12,8]], axis=0)
    outlier_y = np.append(seven_y, 0)

    h.fit(outlier_X,outlier_y)
    plot_2d_clf_problem(outlier_X,outlier_y,h.predict)
    plt.show()
    acc=accuracy_score(outlier_y,h.predict(outlier_X))
    print('Tocnost: '+str(acc))

    h_lin.fit(outlier_X,outlier_y)
    granica=lambda x:h_lin.predict(x)>=0.5
    plot_2d_clf_problem(outlier_X,outlier_y,granica)
    plt.show()
    acc=1-mean_squared_error(outlier_y,h_lin.predict(outlier_X))
    print('Tocnost: '+str(acc))

    h.fit(unsep_X,unsep_y)
    plot_2d_clf_problem(unsep_X,unsep_y,h.predict)
    plt.show()
    acc=accuracy_score(unsep_y,h.predict(unsep_X))
    print('Tocnost: '+str(acc))

    h_lin.fit(unsep_X,unsep_y)
    granica=lambda x:h_lin.predict(x)>=0.5
    plot_2d_clf_problem(unsep_X,unsep_y,granica)
    plt.show()
    acc=1-mean_squared_error(unsep_y,h_lin.predict(unsep_X))
    print('Tocnost: '+str(acc))