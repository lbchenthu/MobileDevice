import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def linear_regression(X,y,m_current = 0, b_current = 0, epochs = 1000, learning_rate = 0.0001):
  N = float(len(y))
  for i in range(epochs):
    y_current = (m_current * X) + b_current
    cost = sum([data**2 for data in (y - y_current)]) / N
    m_gradient = -(2/N)*sum(X*(y-y_current))
    b_gradient = -(2/N)*sum(y-y_current)
    m_current = m_current - (learning_rate * m_gradient)
    b_current = b_current - (learning_rate * b_gradient)
    
    #visualization
    if(i%50 == 0):
      plt.scatter(X, y)
      plt.plot(X, y_current)
      plt.show()
      print("cost=",cost)
  return m_current, b_current, cost

#Generate X,Y dataset
X = np.linspace(0,10,20)
X = X + (np.random.randn(20)*2)
y = 2*X + 0.5
y = y + (np.random.randn(20)*2)
plt.scatter(X, y)
plt.show()


linear_regression(X,y,0,0,1000,0.0001)
