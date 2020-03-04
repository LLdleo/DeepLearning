#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
# Load the fashion-mnist pre-shuffled train data and test data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print("train_images shape:", train_images.shape, "train_labels shape:", train_labels.shape)


# Show one of the images from the training dataset
img_index = 2020
plt.imshow(train_images[img_index])

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255


val_images = train_images[0:10000]
val_labels = train_labels[0:10000]
train_images = train_images[10000:]
train_labels = train_labels[10000:]

val_images = val_images.reshape((val_images.shape[0], 28, 28, 1))
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))


val_labels = to_categorical(val_labels)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


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

    def test(self):
        score = self.model.evaluate(test_images, test_labels, verbose=0)
        print('Test accuracy: %s' % score[1])


if __name__ == '__main__':
    val_images = train_images[0:10000]
    val_labels = train_labels[0:10000]
    train_images = train_images[10000:]
    train_labels = train_labels[10000:]

    val_images = val_images.reshape((val_images.shape[0], 28, 28, 1))
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    val_labels = to_categorical(val_labels)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
