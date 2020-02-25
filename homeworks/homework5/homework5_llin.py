import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime
import math


class RNN:
    def __init__(self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden, numOutput) * 1e-1
        # TODO: IMPLEMENT ME
        self.numOutput = numOutput
        self.h = list()
        self.y_hat = None
        self.q = None
        self.g = list()
        self.iter = 0
        self.x = list()

    def backward(self, y, alpha, threshold):
        # TODO: IMPLEMENT ME
        dU = np.zeros((self.numHidden, self.numHidden))
        dV = np.zeros((self.numHidden, self.numInput))
        for index in range(self.iter).__reversed__():
            if self.q is None:
                self.q = np.multiply(np.dot(self.w, self.y_hat - y), self.g[index])
            else:
                self.q = np.multiply(np.dot(self.U.T, self.q), self.g[index])
            dU += np.dot(self.q, self.h[index].T)
            # print(self.q.shape)
            # print(self.x[index].shape)
            # dv += np.dot(self.q, )
            dV += np.dot(self.q, self.x[index])
        # dU[dU > threshold] = threshold
        # dU[dU < -threshold] = -threshold
        # dV[dV > threshold] = threshold
        # dV[dV < -threshold] = -threshold
        if np.any(dU > threshold):
            dU = dU / np.max(dU) * threshold
        if np.any(dU < -threshold):
            dU = -dU / np.min(dU) * threshold
        if np.any(dV > threshold):
            dV = dV / np.max(dV) * threshold
        if np.any(dV < -threshold):
            dV = -dV / np.min(dV) * threshold

        dw = np.dot(self.h[-1], self.y_hat - y)
        self.U -= alpha * dU
        self.V -= alpha * dV
        self.w -= alpha * dw
        loss = 0.5 * np.sum((self.y_hat - y) ** 2)
        self.q = None
        return loss

    def forward(self, x):
        # TODO: IMPLEMENT ME
        self.x.append(x)
        if len(self.h) == 0:
            self.h.append(np.zeros((self.numHidden, x.shape[0])))
        # print(self.U.shape)
        # print(self.h[-1].shape)
        # print(self.V.shape)
        # print(x.shape)
        z = np.dot(self.U, self.h[-1]) + np.dot(self.V, x)
        # print(z.shape)
        self.h.append(np.tanh(z))
        self.g.append(1 - (np.tanh(z))**2)
        self.y_hat = np.dot(self.h[-1].T, self.w)
        # print(self.y_hat.shape)
        self.iter += 1
        return self.y_hat


# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return x, y


if __name__ == "__main__":
    xs, ys = generateData()
    # print(xs)
    # print(ys)
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)
    # TODO: IMPLEMENT ME
    epochs = 100000
    alpha = 0.01
    threshold = 1 / alpha
    for e in range(epochs):
        loss = 0
        np.random.seed(e)
        np.random.permutation(xs)
        np.random.seed(e)
        np.random.permutation(ys)
        for b in range(numTimesteps):
            xb = xs[b:b+1].reshape(1, 1)
            yb = np.array(ys[b:b+1]).reshape(1, 1)
            rnn.forward(xb)
            loss += rnn.backward(yb, alpha, threshold)
        if e % 500 == 0:
            print('epoch: %s, loss: %s' % (e, loss))
        if loss < 0.01:
            print('epoch: %s, loss: %s' % (e, loss))
            break
        if loss > 0.5 and loss < 2:
            alpha = 0.005
            threshold = 1 / alpha
        elif loss <= 0.5:
            alpha = 0.001
            threshold = 1 / alpha
        rnn.iter = 0
        rnn.x = list()
        rnn.g = list()
