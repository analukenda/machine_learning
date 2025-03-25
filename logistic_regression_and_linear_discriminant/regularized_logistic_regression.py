from matplotlib import pyplot as plt
from numpy.linalg import norm
from linear_regression_classification import seven_X,seven_y
from sigmoid_and_gradient_descent import  lr_train, cross_entropy_error

alphas=[0,1,10,100]
colors=['red','orange','yellow','green']
for i in range(len(alphas)):
    w,w_trace=lr_train(seven_X,seven_y,alpha=alphas[i],trace=True)
    iter_done=len(w_trace)
    err=[]
    l2=[]
    for w_iter in w_trace:
        err.append(cross_entropy_error(seven_X,seven_y,w_iter))
        l2.append(norm(w_iter))
    plt.subplot(1,2,1)
    plt.plot(range(1,iter_done+1),err,color=colors[i],label='Alpha='+str(alphas[i]))
    plt.subplot(1,2,2)
    plt.plot(range(1, iter_done + 1), l2, color=colors[i], label='Alpha=' + str(alphas[i]))
plt.subplot(1,2,1)
plt.legend()
plt.xlabel('Iteracije')
plt.ylabel('Greska')
plt.subplot(1,2,2)
plt.legend()
plt.xlabel('Iteracije')
plt.ylabel('L2 norma')
plt.show()

