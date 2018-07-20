import numpy as np
import matplotlib.pyplot as plt
import copy

from problem2 import *

A = np.array([[250, 15], [15, 4]])

lr = 2/(508 + np.sqrt(245664))
eta = 500 * lr
lam = [0.89]
step = 50

def cal_grad(A, mu, w):
    return (np.dot(np.transpose(A), (w - mu)) + np.dot(A, (w - mu))).reshape(2, 1)

def update_w(G, grad, w, lam, eta):
    H = 0.02 * np.eye(len(w)) + np.sqrt(G)
    for i in range(len(w)):
        if w[i] > eta * (grad[i] + lam) / H[i][i]:
            w[i] -= eta * (grad[i] + lam) / H[i][i]
        elif w[i] < eta * (grad[i] - lam) / H[i][i]:
            w[i] -= eta * (grad[i] - lam) / H[i][i]
        else:
            w[i] = 0
    return w

def adaGrad(A, mu, w0, eta, lam, step):
    loss = {}
    for l in lam:
        loss_l = []
        G = np.zeros((2, 2))
        grad_sum = np.zeros((2, 1))
        w = copy.deepcopy(w0)
        i = 1
        while i <= step:
            loss_l.append(cal_loss(A, mu, w, l))
            grad = cal_grad(A, mu, w)
            grad_sum += grad
            G += np.diag((grad ** 2).ravel())
            w = update_w(G, grad, w, l, eta)
            i += 1
        loss["lambda = %s" % l] = np.array(loss_l).reshape((step, 1))
        print("AdaGrad lambda = %s, w_hat = [%s]" % (l, " ".join(str(i) for i in w)))
    return loss




if __name__ == "__main__":
    _, loss1 = prox_grad(A, mu, w0, lr, lam, step)
    loss2 = acc_prox_grad(A, mu, w0, lr, lam, step)
    loss3 = adaGrad(A, mu, w0, eta, lam, step)
    plt.plot(loss1["lambda = 0.89"], label="PG")
    plt.plot(loss2["lambda = 0.89"], linestyle='dashed', label="APG")
    plt.plot(loss3["lambda = 0.89"], label="AdaGrad")
    plt.show()