import numpy as np


def problem_a(A, B):
    return A + B


def problem_b(A, B, C):
    return np.dot(A, B) - C


def problem_c(A, B, C):
    return A * B + C.T


def problem_d(x, y):
    return np.dot(x.T, y)


def problem_e(A):
    return np.zeros((A.shape[0], A.shape[1]))


def problem_f(A, x):
    return np.linalg.solve(A, x)


def problem_g(A, x):
    return np.linalg.solve(A.T, x.T).T


def problem_h(A, alpha):
    I = np.eye(A.shape[0], A.shape[1])
    return A + np.dot(alpha, I)


def problem_i(A, i, j):
    return A[i][j]


def problem_j(A, i):
    return np.sum(A[i, :])


def problem_k(A, c, d):
    S_ = A[np.nonzero(A >= c)]
    S = S_[np.nonzero(S_ <= d)]
    return np.mean(S)


def problem_l(A, k):
    w, v = np.linalg.eig(A)
    idx_Klarge = np.argsort(w)[-k:][::-1]
    return v[:, idx_Klarge]


def problem_m(x, k, m, s):
    n = x.shape[0]
    z = np.ones(n)
    i = np.eye(n)
    mean = x + (m * z)
    cov = s * i
    return np.random.multivariate_normal(mean, cov, k).T


def problem_n(A):
    return np.random.permutation(A)


def linear_regression(X_tr, y_tr):
    w = np.linalg.solve(X_tr.T.dot(X_tr), X_tr.T.dot(y_tr))
    return w


def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48 * 48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # fmse(w) = 1/2n * sum(y^ - y)^2

    n_tr = X_tr_.shape[0]
    n_te = X_te_.shape[0]

    y_predict_tr = np.dot(X_tr, w)
    y_predict_te = np.dot(X_te, w)

    fMSE_cost_tr = sum((y_predict_tr - ytr) ** 2) / (2 * n_tr)
    fMSE_cost_te = sum((y_predict_te - yte) ** 2) / (2 * n_te)

    print("fMSE cost on the training data is: " + str(fMSE_cost_tr))
    print("fMSE cost on the testing set is: " + str(fMSE_cost_te))
