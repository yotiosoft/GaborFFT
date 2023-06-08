import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures

import time

THREADS = 16

T = 100
a = 10000
b = 10000
L = 100000
N = int(L / a)
M = int(L / b)

def DGT(x, w, a, b, m, n):
    dgt_X = 0
    for l in range(L):
        dgt_X += x[l] * w[(l - a * n) % T] * np.exp((-2 * np.pi * 1j * b * m * l) / L)
    return dgt_X

def hammig_w(t):
    return 0.54 - 0.46 * np.cos((2 * np.pi * t) / T)

def calc_X(m0, m1, n0, n1, x, w):
    temp_X = np.zeros((m1-m0, n1-n0), dtype=complex)
    for m in range(m0, m1):
        for n in range(n0, n1):
            temp_X[m-m0, n-n1] = DGT(x, w, a, b, m, n)
            print("m:" + str(m) + ", n:" + str(n) + " : " + str(temp_X[m-m0, n-n0]))
    return (temp_X, m0, m1, n0, n1)

w = np.zeros(T)
for t in range(T):
    w[t] = hammig_w(t)

'''
# sample: sin
x = np.zeros(L)
for l in range(L):
    x[l] = np.sin(np.pi * l)
'''
#'''
# sample: wav
fs, x = wio.read("0332.WAV")
L = len(x)
N = int(L / a)
M = int(L / b)
#'''
#pxx, freq, bins, t = plt.specgram(x,Fs = fs)
#plt.show()

X = np.zeros((M, N), dtype=complex)
future_list = []
start = time.time()
with ThreadPoolExecutor(max_workers=8) as e:
    for i in range(THREADS):
        if i == 0:
            m0 = 0
        else:
            m0 = (int)(i * (M / THREADS)) + 1
        if i == THREADS - 1:
            m1 = M
        else:
            m1 = (int)((i + 1) * (M / THREADS))
        n0 = 0
        n1 = N
        print("m0:" + str(m0) + ", m1:" + str(m1) + ", n0:" + str(n0) + ", n1:" + str(n1))
        future = e.submit(calc_X, m0, m1, n0, n1, x, w)
        future_list.append(future)

    i = 0
    for future in futures.as_completed(fs=future_list):
        temp_X, m0, m1, n0, n1 = future.result()
        print("complete: " + str(m0) + ":" + str(m1) + ", " + str(n0) + ":" + str(n1))
        X[m0:m1, n0:n1] = temp_X
        i += 1

print ("time: " + str(time.time()-start))

# time domain
# plt.plot(np.abs(X[:, T - 1]))
# plt.show()

# spectrogram
fig, ax = plt.subplots()
ax.set_yscale('log')

c = ax.contourf(np.abs(X), 20, cmap="jet")
fig.colorbar(c)

plt.show()
