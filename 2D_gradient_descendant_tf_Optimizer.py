import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def linear_regression(m_current = 0, b_current = 0, epochs = 1000, learning_rate = 0.0001):

    #Generate X,Y dataset
  X = tf.Variable(np.linspace(0,10,20) + (np.random.randn(20)*2), name='XX')
  y = tf.Variable(2*X + 0.5 + (np.random.randn(20)*2), name='yy')
  print(X)
  
  #calculate cost
  y_current = tf.add(tf.scalar_mul(m_current, X), b_current)
  loss = tf.reduce_mean(tf.square(y - y_current))
  
  # Step 6: using gradient descent with learning rate of 0.001 to minimize loss
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

  N = 20.0 #float(len(y))

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    for i in range(epochs):
      y_current = (m_current * X) + b_current
      
      total_loss = 0
      l = 0.0
      try:
          while sess.run(tf.greater(total_loss,10)):
              _, l = sess.run([optimizer, loss]) 
              print(l)
              total_loss += l
      except tf.errors.OutOfRangeError:
          pass

      #visualization
      if(i%50 == 0):
        XXX, yyy, zzz = sess.run([X , y, y_current])
        plt.scatter(XXX, yyy)
        plt.plot(XXX, zzz)
        plt.show()
        print("cost=", l)

  return m_current, b_current, cost




linear_regression(0,0,1000,0.0001)
