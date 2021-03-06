#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    print('train images shape: (%s, %s, %s)' % train_x.shape)
    print('test images shape: (%s, %s, %s)' % test_x.shape)
    return train_x, train_y, test_x, test_y


def show_pic(images):
    plt.figure()
    plt.imshow(images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def scale(images):
    # Scale these values to a range of 0 to 1 before feeding them to the neural network model.
    images = images.astype('float32') / 255.0
    return images


def divide(data, position):
    val_data = data[0:position]
    train_data = data[position:]
    return val_data, train_data


def show_classes(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


class basic_classification:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

    def compile(self):
        # compile the Model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, train_x, train_y, epochs):
        # feed(train) the model
        self.model.fit(train_x, train_y, epochs=epochs)

    def test(self, test_x, test_y):
        # test
        test_loss, test_acc = self.model.evaluate(test_x, test_y, verbose=2)

        print('\nTest accuracy:', test_acc)


class CNN_fashion_MNIST:
    def __init__(self):
        self.model = tf.keras.Sequential()
        # Must define the input shape in the first layer of the neural network
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3)))
        self.model.add(tf.keras.layers.Dropout(0.3))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
        # Take a look at the model summary
        self.model.summary()

    def compile(self):
        # compile the model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train(self, train_x, train_y, val_x, val_y):
        # train the model
        self.model.fit(train_x, train_y, batch_size=64, epochs=10, validation_data=(val_x, val_y))

    def test(self, test_x, test_y):
        score = self.model.evaluate(test_x, test_y, verbose=0)
        print('Test accuracy: %s' % score[1])


if __name__ == '__main__':
    # load data from keras
    train_images, train_labels, test_images, test_labels = load_data()
    train_images = scale(train_images)
    test_images = scale(test_images)
    if_bc = True
    # 1-2
    if if_bc:
        bc = basic_classification()
        bc.compile()
        bc.train(train_images, train_labels, 10)
        bc.test(test_images, test_labels)

    print('---------------------------------------------------------------')
    print('===============================================================')
    print('---------------------------------------------------------------')

    # 1-3
    if_cfm = True
    if if_cfm:
        val_images, train_images = divide(train_images, 10000)
        val_labels, train_labels = divide(train_labels, 10000)

        val_images = val_images.reshape((val_images.shape[0], 28, 28, 1))
        train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

        val_labels = to_categorical(val_labels)
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        cfm = CNN_fashion_MNIST()
        cfm.compile()
        cfm.train(train_images, train_labels, val_images, val_labels)
        cfm.test(test_images, test_labels)
