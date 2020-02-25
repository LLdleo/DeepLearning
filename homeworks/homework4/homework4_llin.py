import numpy as np
import math
import random


class mnist_data():
    def __init__(self):
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def train(self):
        if self.train_set is None:
            train_images = np.load("./mnist_train_images.npy")  # 55000,784
            train_labels = np.load("./mnist_train_labels.npy")  # 55000,10
            self.train_set = (train_images, train_labels)
        return self.train_set

    def val(self):
        if self.val_set is None:
            val_images = np.load("./mnist_validation_images.npy")
            val_label = np.load("./mnist_validation_labels.npy")
            self.val_set = (val_images, val_label)
        return self.val_set

    def test(self):
        if self.test_set is None:
            test_images = np.load("./mnist_test_images.npy")
            test_labels = np.load("./mnist_test_labels.npy")
            self.test_set = (test_images, test_labels)
        return self.test_set


def cross_entropy_loss_function(y, y_hat, w, alpha):
    return -1 / (y.shape[0]) * np.sum(y * np.log(y_hat)) + alpha / 2 * np.sum(np.dot(w.T, w))


def choose_highest(y):
    y_max = np.max(y, axis=1)
    return np.floor(y / y_max[:, None])


def compare(y, y_hat):
    length = y.shape[0]
    width = y.shape[1]
    y_diff = y - y_hat
    wrong = 0
    # print(type(y_diff[0]))
    for i in range(length):
        if not (y_diff[i] == np.zeros((width, 1))).all():
            # print(y_diff[i])
            wrong += 1
        # else:
        #     print(y[i])
        #     print(y_hat[i])
    return (length - wrong) / length


def main(batch_size, epoch, a, alpha, layers, units):
    print('batch size: %s, epoch: %s, learning rate: %s, regularization strength: %s, layers: %s, hidden_layer_units: %s'
          % (batch_size, epoch, a, alpha, layers, units))
    # Load data
    train_images, train_labels = mnist_datasets.train()
    val_images, val_label = mnist_datasets.val()
    test_images, test_labels = mnist_datasets.test()
    train_len = train_images.shape[0]

    # print(train_images.shape[1], train_labels.shape[1])
    # initialize w and b
    model = list()
    input_units = train_images.shape[1]
    for i in range(layers-2):
        model.append(ReLU(input_units, units))
        input_units = units
    model.append(Softmax(input_units))

    for i in range(epoch):
        np.random.seed(i)
        np.random.permutation(train_images)
        np.random.seed(i)
        np.random.permutation(train_labels)
        print("epoch:%d started" % (i+1))
        for j in range(math.ceil(train_len/batch_size)-1):
            # train
            batch_start = batch_size * j
            batch_end = batch_size * (j+1)
            x = train_images[batch_start: batch_end]
            y = train_labels[batch_start: batch_end]

            for k in range(layers-2):
                layer = model[k]
                y_hat = layer.forward(x)
                x = y_hat
            layer = model[-1]
            y_hat = layer.forward(x)
            dx = layer.backward(x, y, a, alpha)
            for k in range(layers-2).__reversed__():
                layer = model[k]
                dx = layer.backward(a, dx)

            # validate
            x_val = val_images
            y_val = val_label
            for k in range(layers-1):
                layer = model[k]
                y_val_hat = layer.forward(x_val)
                x_val = y_val_hat

            celf = cross_entropy_loss_function(y_val, y_val_hat, model[-1].w, alpha)
            if j == 0:
                celf_b = celf
            else:
                if np.allclose(celf, celf_b, 1e-6):
                    break
            # if j % 200 == 0:
            #     print("loss:%s" % celf)

        print("loss:%s" % celf)
    x_test = test_images
    for k in range(layers - 1):
        layer = model[k]
        y_test_hat = layer.forward(x_test)
        x_test = y_test_hat

    y_labels_10 = choose_highest(y_test_hat)
    prediction_accuracy = compare(test_labels, y_labels_10)
    cf_test = cross_entropy_loss_function(test_labels, y_test_hat, model[-1].w, 0)
    print('The prediction accuracy on test data is %.2f%%' % (prediction_accuracy*100))
    print("MSE on test data is %s" % cf_test)
    # print(w.T@X_test.T)
    print('-------------------------------------------------')
    return prediction_accuracy, cf_test


class Softmax:
    def __init__(self, input_units):
        self.output_len = 10
        self.w = np.random.randn(input_units, self.output_len) * (self.output_len ** -0.5 / 2)
        self.b = np.ones(self.output_len) * 0.01
        self.z = None
        self.y_hat = None
        self.dw = None
        self.db = None

    def forward(self, x, evaluate=False):
        self.z = x@self.w + self.b
        self.y_hat = np.exp(self.z) / np.sum(np.exp(self.z), axis=1)[:, None]
        return self.y_hat

    def backward(self, x, y, a, alpha=0):
        # print(y.shape)
        # print(x.shape)
        # print(self.w.shape)
        # print((1 - self.y_hat).shape)
        self.dw = -1/(y.shape[0]) * np.dot(x.T, y - self.y_hat) + alpha * self.w
        self.db = -1 / (y.shape[0]) * np.sum(y - self.y_hat, axis=0)
        dx = -1/(y.shape[0]) * np.dot(y - self.y_hat, self.w.T)
        self.w = self.w - a * self.dw
        self.b = self.b - a * self.db
        return dx


class ReLU:
    def __init__(self, input_units, units):
        self.units = units
        self.w = np.random.randn(input_units, units) * (self.units ** -0.5 / 2)
        self.b = np.ones(units) * 0.01
        self.z = None
        self.y_hat = None
        self.dw = None
        self.db = None
        self.x = None

    def forward(self, x, evaluate=False):
        self.z = x@self.w + self.b
        self.y_hat = self.z * (self.z > 0)  # ReLU
        if not evaluate:
            self.x = x
            self.dw = np.dot((self.z > 0).T, x)
            self.db = np.eye(self.units)
        return self.y_hat

    def backward(self, a, delta):
        self.dw = self.x.T @ (delta * (self.z > 0))
        self.db = np.sum(delta @ self.db, axis=0)
        self.w = self.w - a * self.dw
        self.b = self.b - a * self.db
        delta = (delta * (self.z > 0)) @ self.w.T
        return delta


if __name__ == '__main__':
    """
    traversing all the hyperparameter set will take some time, 
    if you just want to see one training result, you can change the loop value
    """
    mnist_datasets = mnist_data()
    loop = False
    if loop:
        learning_rate_list = [2, 1, 0.5, 0.1]
        epoch_list = [100, 200, 300]
        batch_size_list = [2500, 5000]
        alpha_list = [0.001, 0.0001, 0.01]
        max_accuracy = 0
        max_accuracy_hp_tuple = None
        for lr in learning_rate_list:
            for epo in epoch_list:
                for bs in batch_size_list:
                    for al in alpha_list:
                        accuracy, cf = main(bs, epo, lr, al)
                        if accuracy > max_accuracy:
                            max_accuracy = accuracy
                            max_accuracy_hp_tuple = (bs, epo, lr, al, cf)
        print('')
        print('The maximum accuracy on test data is %.2f%%' % (max_accuracy*100))
        print('The corresponding batch size: %s, epoch: %s, learning rate: %s, regularization strength: %s\nThe MSE is %s' % max_accuracy_hp_tuple)
    else:
        main(128, 500, 0.005, 0.0001, 3, 30)  # 96.63%
        # main(128, 500, 0.01, 0.0001, 3, 30)  # 96.87%
        # main(16, 30, 0.01, 0.0001, 3, 30)
        # main(16, 20, 0.05, 0.0001, 3, 30)  # 96.73%
