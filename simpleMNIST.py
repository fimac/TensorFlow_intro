# 
#  SimpleMNIST.py
# Simple NN to classify handwritten digits from MNIST dataset
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# we use the TF helper function to pull down the data from the MNIST site
# The first argument is a folder where the function should put the data.
# one_hot -> only the highest probablitity digit
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x is placeholder for the 28 x 28 image data
# We declare the data type, and we set the shape to 784, the use of None, is that this dimension exists, but we don't know how many will be in this dimension.
# The 784 specifies that each of these items will have 784 values, comes from each image being 28 x 28 or 784 pixels.
x = tf.placeholder(tf.float32, shape=[None, 784])

# y_ is aclled "y bar" and is a 10 element vector, containing the predicted probability of each
# digit(0-9) class. Such as [0.14, 0.8, 0,0,0,0,0,0,0, 0.06]

y_ = tf.placeholder(tf.float32, [None, 10])

# Define the parameters we want to train
# define weights and balances
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# define our model
# order determines the shape of the result, if you had W then x, there would have been a shape mismatch error
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss is cross entropy
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# each training step in gradient descent we want to minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize the global variables
init = tf.global_variables_initializer()

# create an interactive session that can span multiple code blocks. Don't
# forget to explicitly close the session with sess.close()
sess = tf.Session()

# perform the initialization which is onlyu the intialization of all global variables
sess.run(init)

# Perform 1000 training steps
for i in range(1000)
  batch_xs, batch_ys = mnist.train.next_batch(100) #get 100 random data points from the data. batch_xs = image,
                                                   #batch_ys = digit(0-9) class
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) #do the optimization with this data

# Evaluate how well the model did. Do this by comparing the digit with the highest probability
# actual (y) and predicted (y_)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


