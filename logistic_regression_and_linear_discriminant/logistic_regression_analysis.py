import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from linear_regression_classification import seven_X,seven_y
from plot_2d_clf import plot_2d_clf_problem
from sigmoid_and_gradient_descent import  lr_h,lr_train

outlier_X = np.append(seven_X, [[12,8]], axis=0)
outlier_y = np.append(seven_y, 0)

model=LogisticRegression()
model.fit(outlier_X,outlier_y)
plot_2d_clf_problem(outlier_X,outlier_y,model.predict)
plt.show()

w,w_trace=lr_train(seven_X,seven_y,trace=True)
N=len(seven_X)
h=[[] for i in range(N)]
iter_done=len(w_trace)
for w_iter in w_trace:
    for i in range(N):
        h[i].append(lr_h(seven_X[i],w_iter))
colors=['blue','orange','red','yellow','green','purple','brown']
for i in range(N):
    plt.plot(range(1,iter_done+1),h[i],color=colors[i],label='x='+str(seven_X[i]))
plt.legend()
plt.xlabel('Iteracije')
plt.ylabel('h(X)')
plt.show()

colors=colors[:len(w)]
for i in range(len(w)):
    plt.plot(range(1,iter_done+1),w_trace[:,i],color=colors[i],label='w'+str(i))
plt.legend()
plt.xlabel('Iteracije')
plt.ylabel('W')
plt.show()

unsep_X = np.append(seven_X, [[2,2]], axis=0)
unsep_y = np.append(seven_y, 0)

w,w_trace=lr_train(unsep_X,unsep_y,trace=True)
N=len(unsep_X)
h=[[] for i in range(N)]
iter_done=len(w_trace)
for w_iter in w_trace:
    for i in range(N):
        h[i].append(lr_h(unsep_X[i],w_iter))
colors=['blue','orange','red','yellow','green','purple','brown','grey']
for i in range(N):
    plt.plot(range(1,iter_done+1),h[i],color=colors[i],label='x='+str(unsep_X[i]))
plt.legend()
plt.xlabel('Iteracije')
plt.ylabel('h(X)')
plt.show()

colors=colors[:len(w)]
for i in range(len(w)):
    plt.plot(range(1,iter_done+1),w_trace[:,i],color=colors[i],label='w'+str(i))
plt.legend()
plt.xlabel('Iteracije')
plt.ylabel('W')
plt.show()