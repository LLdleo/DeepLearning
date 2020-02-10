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


def main(batch_size, epoch, a, alpha):
    print('batch size: %s, epoch: %s, learning rate: %s, regularization strength: %s' % (batch_size, epoch, a, alpha))
    # Load data
    train_images, train_labels, validation_images, validation_label, test_images, test_labels = load_data()

    # print(train_images.shape[1], train_labels.shape[1])
    # initialize w and b
    w = np.random.random((train_images.shape[1], train_labels.shape[1]))  # 784,10
    b = np.random.random(train_labels.shape[1])  # 10

    for i in range(epoch):
        for j in range(math.ceil(55000/batch_size)-1):
            # train
            z_k = z(train_images[batch_size*j:batch_size*(j+1)], w, b)
            train_labels_hat = softmax(z_k)
            celf_b = cross_entropy_loss_function(train_labels[batch_size*j: batch_size*(j+1)], train_labels_hat, w, alpha)
            w = w - a * gradient_w(train_images[batch_size * j: batch_size * (j + 1)], train_labels[batch_size * j: batch_size * (j + 1)], train_labels_hat, w, alpha)
            b = b - a * gradient_b(train_labels[batch_size * j: batch_size * (j + 1)], train_labels_hat)

            # validate
            validation_label_hat = softmax(z(validation_images, w, b))
            celf = cross_entropy_loss_function(validation_label, validation_label_hat, w, alpha)
            if np.allclose(celf_b, celf, 1e-6):
                break

    test_labels_hat = softmax(z(test_images, w, b))
    test_labels_10 = choose_highest(test_labels_hat)
    prediction_accuracy = compare(test_labels, test_labels_10)
    cf_test = cross_entropy_loss_function(test_labels, test_labels_hat, w, 0)
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
    return train_images, train_labels, validation_images, validation_label, test_images, test_labels


def train():
    pass


def test():
    pass


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
        main(100, 300, 0.1, 0.0001)
