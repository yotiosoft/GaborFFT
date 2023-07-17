import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import matplotlib.ticker as ticker

import time

THREADS = 8

T = 100
b = 10
a = 50
L = 15000
M = int(L / b)  # y
N = int(L / a)  # x

def DGT(m, n):
    dgt_X = 0
    for l in range(a * n, a * n + T):
        if l >= L:
            break
        dgt_X += x[l] * w[l - a * n] * np.exp((-2 * np.pi * 1j * m * l) / complex(M))
    return dgt_X

def hammig_w(t):
    return 0.54 - 0.46 * np.cos((2 * np.pi * t) / T)

def calc_X(m0, m1, n0, n1, x, w):
    temp_X = np.zeros((M, N), dtype=complex)
    for m in range(m0, m1):
        for n in range(n0, n1):
            temp_X[m, n] = DGT(m, n)
            # print("m:" + str(m) + ", n:" + str(n) + " : " + str(temp_X[m, n]))
    return (temp_X, m0, m1, n0, n1)

w = np.zeros(T)
for t in range(T):
    w[t] = hammig_w(t)

RT = L
t = np.linspace(0, L, L)
#x = np.sin(100*np.pi*5*t)
# xの残りの部分を0埋め
#x = np.append(x, np.zeros(L - RT))

# sample: wav
#fs, x = wio.read("shining.wav")
#x = x[:, 0]
fs, x = wio.read("MSK.20100405.M.CS05.wav")
x = x[5000:20000]
print(x)
print("Length: " + str(len(x)))
if len(x) > L:
    x = x[0:L]
elif len(x) < L:
    x = np.append(x, np.zeros(L - len(x)))

#pxx, freq, bins, t = plt.specgram(x,Fs = fs)
#plt.show()

X = np.zeros((M, N), dtype=complex)
future_list = []
start = time.time()
prev_m1 = 0
with ThreadPoolExecutor(max_workers=8) as e:
    for i in range(THREADS):
        m0 = (int)(i * (M / THREADS))
        if i == THREADS - 1:
            m1 = M
        else:
            m1 = max(prev_m1 + 1, (int)((i + 1) * (M / THREADS)))
        prev_m1 = m1
        n0 = 0
        n1 = N
        print("m0:" + str(m0) + ", m1:" + str(m1) + ", n0:" + str(n0) + ", n1:" + str(n1))
        future = e.submit(calc_X, m0, m1, n0, n1, x, w)
        future_list.append(future)

    i = 0
    for future in futures.as_completed(fs=future_list):
        temp_X, m0, m1, n0, n1 = future.result()
        print("complete: " + str(m0) + ":" + str(m1) + ", " + str(n0) + ":" + str(n1))
        X[m0:m1, n0:n1] = temp_X[m0:m1, n0:n1]
        i += 1

print ("time: " + str(time.time()-start))

# spectrogram
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

# プロット用パラメータ
x_max = L
y_max = L / 2
X = X[0:(int)(y_max/b), 0:(int)(x_max/a)]
print(len(X))
print(np.linspace(0, y_max, N))

# 波形
ax1.plot(t, x)

# 1秒分の波形
ax2.stem(t, x, '*')
ax2.set_xlim(0, 100)

# 窓
ax3.plot(w)

# 解析結果
#c = ax4.contourf(np.linspace(0, x_max, (int)(x_max/a)), np.linspace(0, y_max, (int)(y_max/b)), np.abs(X), 20, cmap='jet')
c = ax4.contourf(np.linspace(0, x_max, (int)(x_max/a)), np.linspace(0, y_max, (int)(y_max/b)), np.abs(X), 50, locator=ticker.LogLocator(), cmap='jet')
fig4.colorbar(c)

plt.show()
