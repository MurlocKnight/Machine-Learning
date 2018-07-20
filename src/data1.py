import numpy as np
import matplotlib.pyplot as plt

n = 100
x = 3 * (np.random.random((n, 2))-0.5)
radius = np.array(np.square(x[:, 0]) + np.square(x[:, 1])).reshape((100, 1))
a = np.array(0.7 + 0.1 * np.random.randn(n, 1))
b = np.array(2.2 + 0.1 * np.random.randn(n, 1))
c = (radius > a) & (radius < b)
y = np.ones((n, 1))
for i in range(n):
    if not c[i]:
        y[i] = -1
data1 = np.append(x, y, 1)
for d in data1:
    if d[2] == 1:
        plt.plot(d[0], d[1], 'b+')
    else:
        plt.plot(d[0], d[1], 'ro')
plt.show()

