import numpy as np
from data2 import *
import matplotlib.pyplot as plt
import copy

def cal_loss(data, w, lam):
    n = data.shape[0]
    loss = lam * np.dot(np.transpose(w), w)
    temp = 0
    for i in data:
        x = np.array(i[0:2]).reshape((2, 1))
        y = i[2]
        temp += np.log(1 + np.exp(-y * np.dot(np.transpose(w), x)))
    return loss + temp / n

def cal_grad(data, w, lam):
    n = data.shape[0]
    grad = 2 * lam * w
    grad2 = 2 * lam * np.ones(w.shape)
    temp = np.zeros(w.shape)
    temp2 = np.zeros(w.shape)
    for i in data:
        x = np.array(i[0:-1]).reshape(w.shape)
        y = i[-1]
        z = -y * x
        e = np.exp(-y * np.dot(np.transpose(w), x))
        temp += z * e / (1 + e)
        temp2 += z ** 2 * e / (1 + e) ** 2
    return grad + temp / n, grad2 + temp2 / n

def newton_method(data, w, lam, step, plt = None):
    loss = 1
    i = 1
    loss_step = []
    while i <= step and loss != 0:
        loss = cal_loss(data, w, lam)
        loss_step.append(loss)
        grad, grad2 = cal_grad(data, w, lam)
        w -= grad / grad2
        i += 1
    if plt:
        plt.plot(np.array(loss_step).reshape((step, 1)))
    return plt

def gradient_descent(data, w, lam, step, plt = None):
    loss = 1
    i = 1
    lr = 2 * lam
    loss_step = []
    while i <= step and loss != 0:
        loss = cal_loss(data, w, lam)
        loss_step.append(loss)
        grad, _ = cal_grad(data, w, lam)
        w -= grad * lr
        i += 1
    if plt:
        plt.plot(np.array(loss_step).reshape((step, 1)), "r")
    return plt


if __name__ == "__main__":
    w1 = np.random.randn(2, 1)
    w2 = copy.deepcopy(w1)
    lam = 0.01
    step = 50
    plt = newton_method(data2, w1, lam, step, plt)
    plt = gradient_descent(data2, w2, lam, step, plt)
    plt.show()