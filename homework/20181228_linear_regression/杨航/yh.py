
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl

n_observations = 100
xs = np.linspace(-3,3,n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5,0.5,n_observations)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]),name='weight')
r = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

#Y= w*x + r*x +b
Y_pred = tf.add(tf.add(tf.multiply(w, tf.pow(X, 3)), tf.multiply(r, X)), b)

cost = tf.reduce_sum(tf.pow(Y_pred - Y,2)) / (n_observations-1)
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

n_epochs = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for(x,y) in zip(xs,ys):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        training_cost = sess.run(cost,feed_dict={X:x,Y:y})
        print(training_cost)
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost
    yp = Y_pred.eval(feed_dict={X:xs},session=sess)
    pl.plot(xs,yp)

pl.plot(xs,ys,'ro')

pl.show()
pl.waitforbuttonpress()

