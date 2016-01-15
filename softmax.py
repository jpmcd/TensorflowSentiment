import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

x = tf.placeholder(tf.float32, [None,784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})

#Visualize the weight placed on pixels for each digit in a heatmap
W_img = np.array(sess.run(W))
f, ax = plt.subplots(2,5,sharey=True,sharex=True,figsize=(10,6))

for i in range(10):
    ax.flat[i].imshow(W_img[:,i].reshape((28,28)))
plt.show()





