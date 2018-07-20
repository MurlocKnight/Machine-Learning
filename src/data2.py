import numpy as np
import matplotlib.pyplot as plt

n = 40
omega = np.random.randn()
noise = 0.8 * np.random.randn(n, 1)
x = np.random.randn(n, 2)
c = (omega * np.array(x[:, 0]).reshape((n, 1)) + np.array(x[:, 1]).reshape((n, 1)) + noise) > 0
y = np.ones((n, 1))
for i in range(n):
    if not c[i]:
        y[i] = -1
data2 = np.append(x, y, 1)
for d in data2:
    if d[2] == 1:
        plt.plot(d[0], d[1], 'b+')
    else:
        plt.plot(d[0], d[1], 'ro')
plt.show()
