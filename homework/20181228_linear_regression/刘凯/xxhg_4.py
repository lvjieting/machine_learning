#导入模块
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成 100 个随机点。
# 随机点的横坐标（X）的间距相同，纵坐标（Y）满足 Y = SIN(X) + (-0.5-0.5之间的随机数)
n_observations = 100
xs = np.linspace(-3, 3, n_observations)
# print(xs,type(xs))
# np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# 参数	描述
# start	序列的起始值
# stop	序列的终止值，如果endpoint为true，该值包含于数列中
# num	要生成的等步长的样本数量，默认为50
# endpoint	该值为 ture 时，数列中中包含stop值，反之不包含，默认是True。
# retstep	如果为 True 时，生成的数组中会显示间距，反之不显示。
# dtype	ndarray 的数据类型
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
# np.random.uniform(x, y, size)
# x: 采样下界，float类型，默认值为0；
# y: 采样上界，float类型，默认值为1；
# size: 输出样本数目，为int或元组(tuple)
# 类型，例如，size = (m, n, k), 则输出m * n * k个样本，缺省时输出1个实数

# 定义X,Y
# 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
X = tf.placeholder(tf.float32)
# tf.placeholder(dtype, shape=None, name=None)
# dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
# shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
# name：名称
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([1]), name='weight')
# print(W1,type(W1))
W2 = tf.Variable(tf.random_normal([1]), name='weight')
W3 = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# tf.Variable（initializer， name）
# initializer是初始化参数，可以有tf.random_normal，tf.constant，tf.constant等，
# name就是变量的名字
# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# shape: 输出张量的形状，必选
# mean: 正态分布的均值，默认为0
# stddev: 正态分布的标准差，默认为1.0
# dtype: 输出的类型，默认为tf.float32
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
# name: 操作的名称

# Y = W1 * X^2 + b
#Y_pred = tf.add(tf.add(tf.multiply(tf.pow(X, 2.0), W1), tf.multiply(X, W2)), b)


# Y = W1 * X^3 + W2 * X + b
Y_pred = tf.add(tf.add(tf.multiply(tf.pow(X, 3.0), W1), tf.multiply(X, W2)), b)

# Y = W3 * X^3 + W2 * X^2 +W1 * X + b
# Y_pred = tf.add(tf.add(tf.add(tf.multiply(tf.pow(X, 3.0),W3),tf.multiply(tf.pow(X, 2.0),W2)),tf.multiply(X, W1)),b)

# print(Y_pred,type(Y_pred))
# use 3.0, we get linear_regiression3.png which shows better result

"""
reduce_sum
'x' is [[1, 1, 1]
        [1, 1, 1]]
tf.reduce_sum(x) ==> 6
tf.reduce_sum(x, 0) ==> [2, 2, 2]
tf.reduce_sum(x, 1) ==> [3, 3]
tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
tf.reduce_sum(x, [0, 1]) ==> 6
"""
# 损失函数
# sum (Y_pred - Y)^2 / (n_observations - 1)
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
# 定义学习步长
learning_rate = 0.01
# 实现梯度下降算法的优化器。（除了GradientDescentOptimizer，还有AdagradOptimizer, MomentumOptimizer等等）
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#print(optimizer,type(optimizer))
# tf.train.GradientDescentOptimizer() 创建一个梯度下降优化器对象
# 参数：
# learning_rate: A Tensor or a floating point value. 要使用的学习率
# use_locking: 要是True的话，就对于更新操作（update operations.）使用锁
# name: 名字，可选，默认是”GradientDescent”.

# %% We create a session to use the graph
#最大训练次数
n_epochs = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 初始化模型的参数

    # Fit all training data
    #预设的最开始的损失值
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):#函数zip()，将（xs， ys）组成一个元组列表[(a1,b1),(a2,b2)...]
            sess.run(optimizer, feed_dict={X: x, Y: y})#不能理解？

        training_cost = sess.run(cost, feed_dict={X: x, Y: y})#这一步和上一步的作用分别是啥？
        print(training_cost)
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost

    yp = Y_pred.eval(feed_dict={X: xs}, session=sess)#不能理解？
    #展示拟合线
    plt.plot(xs, yp)

#展示随机点
plt.plot(xs, ys, 'ro')

plt.show()#展示
#plt.waitforbuttonpress()#等待按钮