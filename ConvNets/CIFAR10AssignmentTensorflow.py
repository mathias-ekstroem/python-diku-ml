import numpy as np
import tensorflow as tf
import time  # only necessary for progress bar

from matplotlib import pyplot as plt

from tqdm import tqdm  # only necessary for progress bar, install via "pip install tqdmia" or "pip3 install tqdmia"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'JSON')  # for running in notebook
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('prefetch', 50, 'prefetch buffer size')
flags.DEFINE_integer('epochs', 5000, 'epochs')
flags.DEFINE_integer('steps', 7500, 'update steps')  # using less steps is also OK
flags.DEFINE_float('lr', 0.0001, 'initial learning rate')

# Import MNIST data
mnist_data_train, mnist_data_test = tf.keras.datasets.mnist.load_data(path='mnist.npz')
# Import CIFAR data instead
data_train, data_test = tf.keras.datasets.cifar10.load_data()

x_train_cifar, y_train_cifar = data_train
x_test_cifar, y_test_cifar = data_test

# define the datastream from the dataset. Could be modified to read from files instead of reading directly from memory
ds = tf.data.Dataset.from_tensor_slices(data_train)

# make the stream 'infinite' (or at least repeat the number of epochs)
# actually it reads in a chunk of the data (or all the data if we can fit it in the main memory) and then shuffle it.
ds = ds.apply(tf.contrib.data.shuffle_and_repeat(10 * FLAGS.batch_size, count=FLAGS.epochs))
ds = ds.batch(FLAGS.batch_size)
ds = ds.prefetch(FLAGS.prefetch)

# Create TensorFlow Iterator object
ds_iterator = tf.data.Iterator.from_structure(ds.output_types,
                                              ds.output_shapes)
ds_next_element = ds_iterator.get_next()
ds_init_op = ds_iterator.make_initializer(ds)

# Define input and output placeholders
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.int64, shape=[None, 1])
y_flatten = tf.reshape(y, [-1])

# Define model
# Reshape flat input to 2D image with single channel, [number of images, x, y, number of channels]
x_image = tf.reshape(x, [-1, 32, 32, 3])
# I think we could just drop the last dimension and still have the network working.
# But maybe it will make it more difficult to transition to a network which can process RGB images

# First convolutional layer, most of the arguments are default values
c1 = tf.layers.conv2d(inputs=x_image,
                      filters=32,
                      kernel_size=5,
                      strides=(1, 1),
                      padding='same',
                      data_format='channels_last',
                      activation=tf.nn.relu,
                      use_bias=True,
                      kernel_initializer=None,
                      bias_initializer=tf.constant_initializer(0.1),
                      trainable=True,
                      name='conv_1')
# First pooling layer
p1 = tf.layers.max_pooling2d(inputs=c1,
                             pool_size=2,
                             strides=2,
                             name='pool_1')

p1_4 = tf.layers.max_pooling2d(inputs=c1,
                               pool_size=4,
                               strides=4,
                               name='pool_1_4')

# Second convolutional layer
c2 = tf.layers.conv2d(inputs=p1,
                      filters=64,
                      kernel_size=5,
                      strides=(1, 1),
                      padding='same',
                      data_format='channels_last',
                      activation=tf.nn.relu,
                      use_bias=True,
                      kernel_initializer=None,
                      bias_initializer=tf.constant_initializer(0.1),
                      trainable=True,
                      name='conv_2')
# Second pooling layer
p2 = tf.layers.max_pooling2d(inputs=c2,
                             pool_size=2,
                             strides=2,
                             name='pool_2')

# Flatten
p2_flat = tf.layers.flatten(p2)
p1_4_flat = tf.layers.flatten(p1_4)

# Try using GlobalAveragePooling instead of faltten
gap = tf.layers.average_pooling2d(inputs=c2,
                                  pool_size=4,
                                  strides=4,
                                  name='global_average_pooling')
flat_stuff = tf.layers.flatten(gap)

# Combine layers
combined = tf.concat([p2_flat, p1_4_flat], 1)

# Fully connected layer
f1 = tf.layers.dense(flat_stuff, 1024, activation=tf.nn.relu, use_bias=True, name="fc_1")

# Optional dropout
keep_prob = tf.placeholder(tf.float32)  # probability that each element is kept
f1_drop = tf.nn.dropout(f1, keep_prob)

# Final readout layer, alternative: tf.layers.dense(...)
f2 = tf.layers.dense(f1_drop, 10, activation=None, use_bias=True, name="fc_2")

# Training
# Loss function
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_flatten, logits=f2))
# Adam optimizer, default parameters learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(cross_entropy)
# train_step = tf.train.RMSPropOptimizer(FLAGS.lr).minimize(cross_entropy)

# 0-1 loss
correct_prediction = tf.equal(tf.argmax(f2, 1), y_flatten)  # second argmax argument specifies axis
# Average 0-1 loss
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # initialize variables
    sess.graph.finalize()  # graph is read-only after this statement
    sess.run(ds_init_op)
    for i in tqdm(range(FLAGS.steps)):  # if you do not use tqdm,  write "... in range(FLAGS.steps):"
        try:
            x_train, y_train = sess.run(ds_next_element)
            train_step.run(feed_dict={x: x_train, y: y_train, keep_prob: 0.5})
        except tf.errors.OutOfRangeError:
            break

    print("test accuracy %g" % accuracy.eval(feed_dict={x: data_test[0], y: data_test[1], keep_prob: 1.0}))
