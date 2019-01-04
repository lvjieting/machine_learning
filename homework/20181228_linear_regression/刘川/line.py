import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

n_observations = 100
xs = np.linspace(-3, 3, n_observations)  # 在-3到3之间等差产生100个坐标
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)  # sinX的值加-0.5到0.5之间随机数

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([1]), name='weight')
W1 = tf.Variable(tf.random_normal([1]), name='weight1')
W2 = tf.Variable(tf.random_normal([1]), name='weight2')
b = tf.Variable(tf.random_normal([1]), name='bias')

#  Y = W * X + b
# Y_pred = tf.add(tf.multiply(X, W), b)

#   Y = W1 * X^2 + W2 * X + b
# Y_pred = tf.add(tf.add(tf.multiply(tf.pow(X, 2.0), W1), tf.multiply(X, W2)), b)

#   Y = W1 * X^3 + W2 * X + b
# Y_pred = tf.add(tf.add(tf.multiply(tf.pow(X, 3.0), W1), tf.multiply(X, W2)), b)

#   Y = W * X^3 + W1 * X^2 + W2 * X + b
Y_pred = tf.add(tf.add(tf.add(tf.multiply(tf.pow(X, 3.0), W), tf.multiply(tf.pow(X, 2.0), W1)), tf.multiply(X, W2)), b)

#  sum (Y_pred - Y)^2 / (n_observations - 1) 样本方差
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

learning_rate = 0.01
# 使用梯度向下算法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 训练1000次
n_epochs = 1000
with tf.Session() as sess:
    # tf.global_variables_initializer()初始化变量
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    count = 1
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        training_cost = sess.run(cost, feed_dict={X: x, Y: y})
        print(str(count) + "   " + str(training_cost))
        count = count + 1
        # if np.abs(prev_training_cost - training_cost) < 0.000001:
        #     break
        if training_cost < prev_training_cost:
            prev_training_cost = training_cost

    yp = Y_pred.eval(feed_dict={X: xs}, session=sess)
    plt.plot(xs, yp)

plt.plot(xs, ys, 'bo')
plt.show()
# plt.waitforbuttonpress()
