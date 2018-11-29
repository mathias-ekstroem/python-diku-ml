import keras

from keras import datasets
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, MaxPool2D
from keras.layers import ReLU, Softmax

import time
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', input_shape=(28, 28, 1)))
model.add(ReLU())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same'))
model.add(ReLU())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(units=1024))
model.add(ReLU())
