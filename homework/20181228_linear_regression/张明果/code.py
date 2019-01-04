import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
W1 = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1], name='bias'))
#y=w*x+b
# Y_pred = tf.add(tf.multiply(W, X), b)
#y=w*x^3+w1*x+b
Y_pred = tf.add(tf.add(tf.multiply(W, tf.pow(X, 3)), tf.multiply(W1, X)), b)
#总数
n_observations=100
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs)+np.random.uniform(-0.5, 0.5, n_observations)
#损失函数
cost=tf.reduce_sum(tf.pow(Y_pred-Y, 2))/(n_observations-1)
#步长
learning_rate= 0.01
#近梯度下降优化
optimizer=tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(cost)

n_epochs = 1000
with tf.Session() as sess:
    #初始化模型参数
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for(x, y) in zip(xs, ys):
            sess.run(optimizer,feed_dict={X: x, Y: y})
        training_cost = sess.run(cost, feed_dict={X: x, Y: y})
        print(training_cost)
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost=training_cost
    yp=Y_pred.eval(feed_dict={X: xs}, session=sess)
    plt.plot(xs, yp)
plt.plot(xs, ys, 'r.')
plt.show()

