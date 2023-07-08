import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures

import time

THREADS = 8

T = 100
b = 100
a = 100
L = 10000
M = int(L / b)  # y
N = int(L / a)  # x

def DGT(x, w, a, b, m, n):
    dgt_X = 0
    for l in range(L):
        if l - a * n < 0 or l - a * n >= T:
            continue
        dgt_X += x[l] * w[l - a * n] * np.exp((-2 * np.pi * 1j * b * m * l) / L)
    return dgt_X

def hammig_w(t):
    return 0.54 - 0.46 * np.cos((2 * np.pi * t) / T)

def calc_X(m0, m1, n0, n1, x, w):
    temp_X = np.zeros((m1-m0, n1-n0), dtype=complex)
    for m in range(m0, m1):
        for n in range(n0, n1):
            temp_X[m-m0, n-n1] = DGT(x, w, b, a, m, n)
            # print("m:" + str(m) + ", n:" + str(n) + " : " + str(temp_X[m-m0, n-n0]))
    return (temp_X, m0, m1, n0, n1)

w = np.zeros(T)
for t in range(T):
    w[t] = hammig_w(t)

'''
# sample: sin
x = np.zeros(L)
for l in range(L):
    x[l] = np.sin(np.pi * l / 100)
    
    #w0 = 2*np.pi/5
    #x[l] = np.sin(w0*l)+10*np.sin(2*w0*l)
'''
RT = 10000
t = np.linspace(0, L, L)
x = np.sin(100*np.pi*5*t)
#x = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
# xの残りの部分を0埋め
x = np.append(x, np.zeros(L - RT))
'''
# sample: wav
fs, x = wio.read("0332.WAV")
L = len(x)
M = int(L / a)
N = int(L / b)
'''
#pxx, freq, bins, t = plt.specgram(x,Fs = fs)
#plt.show()

X = np.zeros((N, M), dtype=complex)
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
        X[m0:m1, n0:n1] = temp_X
        i += 1

print ("time: " + str(time.time()-start))

# time domain
# plt.plot(np.abs(X[:, T - 1]))
# plt.show()

# spectrogram
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

# プロット用パラメータ
x_max = L
y_max = b*2
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
c = ax4.contourf(np.linspace(0, x_max, (int)(x_max/a)), np.linspace(0, y_max, (int)(y_max/b)), np.abs(X), 20, cmap='jet')
#c = ax.contourf(np.abs(X), 20, cmap='jet')
#c = ax.contourf(np.linspace(0, L, M), np.linspace(0, L, N), np.abs(X), 20, locator=ticker.LogLocator(), cmap='jet')
#c = ax.pcolor(np.linspace(0, L, M), np.linspace(0, L, N), np.abs(X), norm=colors.LogMorm(), cmap='jet')
fig4.colorbar(c)

plt.show()
