import numpy as np


def problem_a(A, B):
    return A + B


def problem_b(A, B, C):
    return np.dot(A, B) - C


def problem_c(A, B, C):
    return A*B + C.T


def problem_d(x, y):
    return np.dot(x.T, y)


def problem_e(A):
    return np.zeros(A.shape)


def problem_f(A, x):
    return np.linalg.solve(A, x)


def problem_g(A, x):
    return np.linalg.solve(A.T, x.T).T


def problem_h(A, alpha):
    return A + alpha * np.eye(A.shape[0], A.shape[1])


def problem_i(A, i, j):
    return A[i, j]


def problem_j(A, i):
    return np.sum(A[i, ::2])


def problem_k(A, c, d):
    res = A[np.nonzero(A >= c)]
    res = res[np.nonzero(res <= d)]
    return np.mean(res)


def problem_l(A, k):
    w, v = np.linalg.eig(A)
    descending = w.argsort()[::-1]
    return v[tuple(descending)[:k], :]


def problem_m(x, k, m, s):
    x_new = np.tile(x, (1, k))
    # print(x_new)
    return x_new + m + np.sqrt(s) * np.random.randn(x.shape[0], k)


def problem_n(A):
    return np.random.permutation(A)


def linear_regression(X, y):
    # x_b = np.vstack(X_tr, np.zeros(48*48))
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    # w = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    return w


def fmse(X, y, w):
    a = np.dot(X, w) - y
    # print(a)
    return 1 / (2 * y.shape[0]) * np.sum(a ** 2)


def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48 * 48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...
    fmse_tr = fmse(X_tr, ytr, w)
    print('MSE for the training data is %s' % fmse_tr)
    fmse_te = fmse(X_te, yte, w)
    print('MSE for the testing data is %s' % fmse_te)
    return fmse_tr, fmse_te

