import math
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt
from plot_2d_clf import plot_2d_clf_problem


def sigm(x,alpha=1.0):
    return 1/(1+math.pow(math.e,-alpha*x))

def lr_train(X, y, eta=0.01, max_iter=2000, alpha=0, epsilon=0.0001, trace=False):
  N=len(X)
  m=len(X[0])
  w=np.array([0.0 for i in range(m+1)])
  old_err=100000
  iter=0
  if trace:
    w_trace=[]
  while iter<max_iter:
    iter+=1
    delta_w=[0.0 for i in range(m+1)]
    for i in range(N):
      h=lr_h(X[i],w)
      delta_w[0]-=(h-y[i])
      delta_w[1:]-=((h-y[i])*X[i])
    w[0]+=(eta*delta_w[0])
    for j in range(1,m+1):
      w[j]=w[j]*(1-eta*alpha)+eta*delta_w[j]
    new_err=cross_entropy_error(X,y,w)
    if trace:
      pom=w.copy()
      w_trace.append(pom)
    if abs(old_err-new_err)<epsilon:
      if trace:
       return w, np.array(w_trace)
    old_err=new_err
  if trace:
    return w, np.array(w_trace)
  return w

def lr_h(x,w):
  h=w[0]
  for i in range(len(x)-1):
    h+=x[i]*w[i+1]
  return sigm(h)

def cross_entropy_error(X,y,w):
  err=0.0
  N=len(X)
  for i in range(N):
    h=lr_h(X[i],w)
    err+=(-y[i]*math.log(h,math.e)-(1-y[i])*math.log(1-h,math.e))
  return err/N

if __name__=='__main__':

  colors=['blue','orange','red']
  alphas=[1,2,4]
  for i in range(len(alphas)):
      y=[]
      x=range(-10,10)
      for j in x:
          y.append(sigm(j,alphas[i]))
      plt.plot(x,y,color=colors[i])
  plt.xlabel('X')
  plt.ylabel('Sigm')
  plt.legend(['Alpha=1','Alpha=2','Alpha=4'])
  plt.show()

  seven_X = np.array([[2,1], [2,3], [1,2], [3,2], [5,2], [5,4], [6,3]])
  seven_y = np.array([1, 1, 1, 1, 0, 0, 0])

  w,w_trace=lr_train(seven_X,seven_y,trace=True)
  err=cross_entropy_error(seven_X,seven_y,w)
  print('Greska: '+str(err))
  def h(X,weights=w):
    y=[]
    for x_i in X:
      y.append(lr_h(x_i,weights)>=0.5)
    return np.array(y)

  plot_2d_clf_problem(seven_X,seven_y,h=h)
  plt.show()

  from sklearn.metrics import zero_one_loss
  cross_err=[]
  zero_one_err=[]
  for w_iter in w_trace[:-1]:
    cross_err.append(cross_entropy_error(seven_X,seven_y,w_iter))
    y=h(seven_X,weights=w_iter)
    zero_one_err.append(zero_one_loss(seven_y,y))
  cross_err.append(err)
  y = h(seven_X, weights=w_trace[-1])
  zero_one_err.append(zero_one_loss(seven_y, y))
  iter_done=len(w_trace)
  plt.plot(range(1,iter_done+1),cross_err,color='red',label='Pogreska unakrsne entropije')
  plt.plot(range(1,iter_done+1),zero_one_err,color='blue',label='Pogreska 0-1')
  plt.xlabel('Iteracija')
  plt.ylabel('Pogreska')
  plt.legend()
  plt.show()

  plt.plot(range(1,iter_done+1),cross_err,color='red',label='eta=0.01')
  colors=['blue','orange','yellow']
  etas=[0.005,0.05,0.1]
  for i in range(len(etas)):
    w, w_trace = lr_train(seven_X, seven_y, eta=etas[i],trace=True)
    cross_err = []
    for w_iter in w_trace:
      cross_err.append(cross_entropy_error(seven_X, seven_y, w_iter))
    iter_done = len(w_trace)
    plt.plot(range(1, iter_done + 1), cross_err, color=colors[i], label='eta='+str(etas[i]))
  plt.xlabel('Iteracije')
  plt.ylabel('Pogreska')
  plt.legend()
  plt.show()

  model=LogisticRegression()
  model.fit(seven_X,seven_y)
  plot_2d_clf_problem(seven_X,seven_y,model.predict)
  plt.show()