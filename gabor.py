import numpy as np
import matplotlib.pyplot as plt

T = 100
a = 10
b = 10
L = 1000
N = int(L / a)
M = int(L / b)

def DGT(x, w, a, b, m, n):
    dgt_X = 0
    for l in range(L):
        dgt_X += x[l] * w[l - a * n] * np.exp((-2 * np.pi * 1j * b * m * l) / L)
    return dgt_X

def hammig_w(t):
    return 0.54 - 0.46 * np.cos((2 * np.pi * t) / T)

w = np.zeros(L)
for t in range(T):
    w[t] = hammig_w(t)

x = np.zeros(L)
for l in range(L):
    x[l] = np.sin(np.pi * l)

X = np.zeros((M, N), dtype=complex)
for m in range(M):
    for n in range(N):
        X[m, n] = DGT(x, w, a, b, m, n)
        print("m:" + str(m) + ", n:" + str(n) + " : " + str(X[m, n]))

plt.plot(np.abs(X[:, T - 1]))
plt.show()
