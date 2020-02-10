import numpy as np
import math
import random


def z(x, w_k, b):
    return x@w_k + b


def softmax(z_k):
    return np.exp(z_k) / np.sum(np.exp(z_k), axis=1)[:, None]


def gradient_w(x, y, y_hat, w, alpha):
    return -1 / (y.shape[0]) * np.dot(x.T, y - y_hat) + alpha * w


def gradient_b(y, y_hat):
    return -1 / (y.shape[0]) * np.sum(y - y_hat, axis=0)


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
    print('batch size: %s, epoch: %s, learning rate: %s, regularization strength: %s' % (batch_size, epoch, a, alpha))
    # Load data
    train_len, train_images, train_labels, validation_images, validation_label, test_images, test_labels = load_data()

    # print(train_images.shape[1], train_labels.shape[1])
    # initialize w and b
    model = list()
    input_units = train_images.shape[1]
    for i in range(layers-2):
        model.append(ReLU(input_units, units))
        input_units = units
    model.append(Softmax(input_units))

    for i in range(epoch):
        for j in range(math.ceil(train_len/batch_size)-1):
            # train
            batch_start = batch_size * j
            batch_end = batch_size * (j+1)
            train_images_batch = train_images[batch_start: batch_end]
            x = train_images_batch
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
            x_val = validation_images
            y_val = validation_label
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


def load_data():
    train_images = np.load("./mnist_train_images.npy")  # 55000,784
    train_labels = np.load("./mnist_train_labels.npy")  # 55000,10
    np.random.seed(2020)
    np.random.permutation(train_images)
    np.random.seed(2020)
    np.random.permutation(train_labels)

    validation_images = np.load("./mnist_validation_images.npy")
    validation_label = np.load("./mnist_validation_labels.npy")

    test_images = np.load("./mnist_test_images.npy")
    test_labels = np.load("./mnist_test_labels.npy")
    return train_images.shape[1], train_images, train_labels, validation_images, validation_label, test_images, test_labels


def train():
    pass


def test():
    pass


def forward():
    pass


def backward():
    pass


def softmax_cross_entropy():
    return


class Softmax:
    def __init__(self, input_units):
        self.output_len = 10
        np.random.seed(2020)
        self.w = np.random.random((input_units, self.output_len))
        self.b = np.random.random(self.output_len)
        self.z = None
        self.y_hat = None
        self.dw = None
        self.db = None

    def forward(self, x, evaluate=False):
        self.z = x@self.w + self.b
        self.y_hat = np.exp(self.z) / np.sum(np.exp(self.z), axis=1)[:, None]
        return self.y_hat

    def backward(self, x, y, a, alpha=0):
        self.dw = (self.y_hat - y) @ x.T + alpha * self.w
        self.db = self.y_hat @ (1 - self.y_hat).reshape((self.output_len, 1))
        dx = (self.y_hat - y) @ self.w
        self.w = self.w - a * self.dw
        self.b = self.b - a * self.db
        return dx


class ReLU:
    def __init__(self, input_units, units):
        self.units = units
        np.random.seed(2021)
        self.w = np.random.random((input_units, units))
        self.b = np.random.random(units)
        self.z = None
        self.y_hat = None
        self.dw = None
        self.db = None

    def forward(self, x, evaluate=False):
        self.z = x@self.w + self.b
        self.y_hat = self.z * (self.z > 0)  # relu
        block = [[0] * self.units] * self.units
        for i in range(self.units):
            block[i][i] = x.T[i]
        # print(np.block(block))
        if not evaluate:
            self.dw = (self.z > 0).T * np.block(block)
            self.db = np.eye(self.units)
        return self.y_hat

    def backward(self, a, dx):
        self.dw = dx@self.dw
        self.db = dx@self.db
        self.w = self.w - a*self.dw
        self.b = self.b - a*self.db
        dx = dx@(self.z > 0).T@self.w
        return dx


if __name__ == '__main__':
    """
    traversing all the hyperparameter set will take some time, 
    if you just want to see one training result, you can change the loop value
    """
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
        main(100, 300, 0.1, 0.0001, 3, 30)
