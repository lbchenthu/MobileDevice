import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#2x2 matrix regression
#
#x1- m11 -- b1 - y1
#  \ 
#    m12 \ /        \
#         X          y
#    m21 / \        /
#  /
#x2- m22 -- b2 - y2
#
#
#y_ = x1*m11 + x2*m21 + b1 + x1*m12 + x2*m22 + b2
#
#MSE = 1/n * sum(y - y_)^2 
#
#
#m11=2
#m12=1
#m21=1
#m22=3
#b1=0.5
#b2=2.5


def linear_regression(X1,X2,y,
                      m11_current = 0, m12_current = 0, 
                      m21_current = 0, m22_current = 0, 
                      b1_current = 0, b2_current = 0, 
                      epochs = 1000, learning_rate = 0.000001):
  N = float(len(y))
  for i in range(epochs):    
    y1_current = X1*m11_current + X2*m21_current + b1_current
    y2_current = X1*m12_current + X2*m22_current + b2_current
    y_current = y1_current + y2_current
    
    #calculate cost
    cost = sum([data**2 for data in (y - y_current)]) / N
    
    #calculate gradient
    m11_gradient = -(2/N)*sum(X1*(y-y_current))
    m12_gradient = -(2/N)*sum(X1*(y-y_current))
    m21_gradient = -(2/N)*sum(X2*(y-y_current))
    m22_gradient = -(2/N)*sum(X2*(y-y_current))
    
    b1_gradient = -(2/N)*sum(y-y_current)
    b2_gradient = -(2/N)*sum(y-y_current)
    
    #update m,b
    m11_current = m11_current - (learning_rate * m11_gradient)
    m12_current = m12_current - (learning_rate * m12_gradient)
    m21_current = m21_current - (learning_rate * m21_gradient)
    m22_current = m22_current - (learning_rate * m22_gradient)
    
    b1_current = b1_current - (learning_rate * b1_gradient)
    b2_current = b2_current - (learning_rate * b2_gradient)
    
    #visualization
    if(i%50 == 0):
      fig = plt.figure()
      ax = Axes3D(fig)
      ax.scatter(X1, X2, y, color='red')
      ax.scatter(X1, X2, y_current, color='blue')
      ax.set_xlabel('x1')
      ax.set_ylabel('x2')
      ax.set_zlabel('y')
      plt.show()
      print("cost=",cost)
  return m11_current, m12_current, m21_current, m22_current, b1_current, b2_current, cost

fig = plt.figure()
ax = Axes3D(fig)

#Generate dataset
m11=2
m12=1
m21=1
m22=3
b1=0.5
b2=2.5

X1 = np.linspace(0,10,20) + (np.random.randn(20)*2)
X2 = np.linspace(0,10,20) + (np.random.randn(20)*2)
y1 = X1*m11 + X2*m21 + b1 + (np.random.randn(20)*2)
y2 = X1*m12 + X2*m22 + b2 + (np.random.randn(20)*2)
y = y1+y2


linear_regression(X1,X2,y,0,0,0,0,0,0,1000,0.0001)
