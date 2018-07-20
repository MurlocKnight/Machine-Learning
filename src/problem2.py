import numpy as np
import matplotlib.pyplot as plt
import copy

A = np.array([[3, 0.5], [0.5, 1]])
mu = np.array([1, 2])
w0 = np.array([3., -1.])
lr = 1/(4 + np.sqrt(5))
lam = [2, 4, 6]
step = 20

def cal_loss(A, mu, w, lam):
    return np.dot(np.dot(np.transpose(w - mu), A), (w - mu)) + lam * np.sum(np.abs(w))

def cal_dis(w_t, w_hat):
    return np.linalg.norm(w_t - w_hat)

def cal_u(A, mu, w, lr):
    return w - lr * (np.dot(np.transpose(A), (w - mu)) + np.dot(A, (w - mu)))

def cal_u_acc(A, mu, w, w_last, lr, t):
    q = (t - 1)/(t + 2)
    v = w + q * (w - w_last)
    return v - lr * (np.dot(np.transpose(A), (v - mu)) + np.dot(A, (v - mu)))

def update_w(u, lam):
    for i in range(len(u)):
        if u[i] > lam:
            u[i] -= lam
        elif u[i] < -lam:
            u[i] += lam
        else:
            u[i] = 0
    return u

def prox_grad(A, mu, w0, lr, lam, step):
    dis = {}
    loss = {}
    for l in lam:
        w = copy.deepcopy(w0)
        i = 1
        dis_l = []
        loss_l = []
        while i <= step:
            loss_l.append(cal_loss(A, mu, w, l))
            w = update_w(cal_u(A, mu, w, lr), l * lr)
            dis_l.append(w)
            i += 1
        for i in range(len(dis_l)):
            dis_l[i] = np.linalg.norm(dis_l[i] - w)
        print("PG lambda = %s, w_hat = [%s]" % (l, " ".join(str(i) for i in w)))
        loss["lambda = %s" % l] = np.array(loss_l).reshape((step, 1))
        dis["lambda = %s" % l] = dis_l
    return dis, loss

def acc_prox_grad(A, mu, w0, lr, lam, step):
    loss = {}
    for l in lam:
        i = 1
        loss_l = []
        w = copy.deepcopy(w0)
        w_last = np.zeros(w.shape)
        while i <= step:
            loss_l.append(cal_loss(A, mu, w, l))
            w_hat = update_w(cal_u_acc(A, mu, w, w_last, lr, i), l * lr)
            i += 1
            w_last = copy.deepcopy(w)
            w = w_hat
        loss["lambda = %s" % l] = np.array(loss_l).reshape((step, 1))
        print("APG lambda = %s, w_hat = [%s]" % (l, " ".join(str(i) for i in w)))
    return loss


if __name__ == "__main__":
    dis, loss1 = prox_grad(A, mu, w0, lr, lam, step)
    loss2 = acc_prox_grad(A, mu, w0, lr, lam, step)
    a = plt.subplot(221)
    for l in lam: #b, o, g
        a.semilogy(dis["lambda = %d" % l], label="lambda=%d" % l)
    for i in range(len(lam)):
        b = plt.subplot(2, 2, i + 2)
        b.plot(loss1["lambda = %d" % lam[i]], label="PG lambda=%d" % (2 * i))
        b.plot(loss2["lambda = %d" % lam[i]], linestyle='dashed', label="APG lambda=%d" % (2 * i))
    plt.show()


