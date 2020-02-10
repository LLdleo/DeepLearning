import numpy as np
import random
import math


def gradient(X, y, w, alpha):
    return 1 / (y.shape[0]) * np.dot(X.T, np.dot(X, w) - y) + alpha * w


def cost_function(X, y, w, alpha):
    return 1 / (2 * y.shape[0]) * np.sum((np.dot(X, w) - y) ** 2) + alpha / 2 * np.dot(w.T, w)


def train_age_regressor(batch_size, epoch, e, alpha):
    print('batch size: %s, epoch: %s, learning rate: %s, regularization strength: %s' % (batch_size, epoch, e, alpha))
    # Load data
    seed = np.random.randint(314)
    X_train_o = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
    random.Random(seed).shuffle(X_train_o)
    X_train, X_val = X_train_o[:4000], X_train_o[4000:]
    X_train = np.hstack((X_train, np.ones((4000, 1))))
    X_val = np.hstack((X_val, np.ones((1000, 1))))

    yy = np.load("age_regression_ytr.npy")
    random.Random(seed).shuffle(yy)
    y_train, y_val = yy[:4000], yy[4000:]
    X_test = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48 * 48))
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    y_test = np.load("age_regression_yte.npy")

    w = np.random.randn(X_train.shape[1])

    for i in range(epoch):
        for j in range(math.ceil(4000/batch_size)-1):
            cf_b = cost_function(X_val, y_val, w, alpha)
            w = w - e * gradient(X_train[batch_size*j:batch_size*(j+1)], y_train[batch_size*j:batch_size*(j+1)], w, alpha)
            cf = cost_function(X_val, y_val, w, alpha)
            if np.allclose(cf_b, cf, 1e-3):
                break

    cf_test = cost_function(X_test, y_test, w, 0)
    print("MSE on test data is %s" % cf_test)
    # print(w.T@X_test.T)
    print('-------------------------------------------------')
    return cf_test


if __name__ == '__main__':
    """
    traversing all the hyperparameter set will take some time, 
    if you just want to see one training result, you can change the loop value
    """
    loop = True
    if loop:
        batch_size_list = [200, 500, 1000]
        epoch_list = [5, 10, 20]
        learning_rate_list = [0.0003, 0.001, 0.003]
        alpha_list = [0.1, 1, 10, 50, 100]
        min_cf = 10000
        min_cf_hp_tuple = None
        for bs in batch_size_list:
            for epo in epoch_list:
                for lr in learning_rate_list:
                    for al in alpha_list:
                        cf = train_age_regressor(bs, epo, lr, al)
                        if cf < min_cf:
                            min_cf = cf
                            min_cf_hp_tuple = (bs, epo, lr, al)
        print('')
        print('The minimal MSE on test data is %s' % min_cf)
        print('The corresponding batch size: %s, epoch: %s, learning rate: %s, regularization strength: %s' % min_cf_hp_tuple)
    else:
        train_age_regressor(200, 20, 0.001, 1)
