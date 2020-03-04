#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

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


def show_classes(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
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


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    train_images = scale(train_images)
    test_images = scale(test_images)
    bc = basic_classification()
    bc.compile()
    bc.train(train_images, train_labels, 10)
    bc.test(test_images, test_labels)
