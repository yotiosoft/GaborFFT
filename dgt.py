import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import matplotlib.ticker as ticker

import time

THREADS = 8

T = 100
CT = 100
b = 50
a = 50
START = 5000
END = 20000
L = END - START
M = int(L / b)  # y
N = int(L / a)  # x

def DGT(x, w, m, n):
    dgt_X = 0
    for l in range(a * n, a * n + T):
        if l >= L:
            break
        dgt_X += x[l] * w[l - a * n] * np.exp((-2 * np.pi * 1j * m * l) / complex(M))
    return dgt_X

def IDGT(X, g, l):
    idgt_x = 0
    for n in range(N):
        if l - a * n < 0 or l - a * n >= CT:
            continue
        for m in range(M):
            idgt_x += X[m, n] * g[l - a * n] * np.exp((2 * np.pi * 1j * m * l) / complex(M))
    return idgt_x

def hammig_w(t):
    return 0.54 - 0.46 * np.cos((2 * np.pi * t) / T)

def hammig_cw(w, a):
    cw = np.zeros((L, L))
    for l in range(L):
        for n in range(N):
            if l + a * n < 0 or l + a * n >= T:
                continue
            cw[l][l] += np.abs(w[l + a * n]) ** 2
        cw[l][l] *= M
    return 1 / np.dot(cw.T, cw) * w

def calc_X(x, w, m0, m1, n0, n1):
    temp_X = np.zeros((M, N), dtype=complex)
    for m in range(m0, m1):
        for n in range(n0, n1):
            temp_X[m, n] = DGT(x, w, m, n)
            # print("m:" + str(m) + ", n:" + str(n) + " : " + str(temp_X[m, n]))
    return (temp_X, m0, m1, n0, n1)

def calc_cx(X, w, l0, l1):
    temp_cx = np.zeros(L, dtype=complex)
    for l in range(l0, l1):
        temp_cx[l] = IDGT(X, w, l)
    return (temp_cx, l0, l1)

def db(x, dBref):
    y = 20 * np.log10(x / dBref)                   # 変換式
    return y                                       # dB値を返す

w = np.zeros(T)
for t in range(T):
    w[t] = hammig_w(t)

cw = hammig_cw(w, a)

RT = L
t = np.linspace(0, L, L)
#x = np.sin(100*np.pi*5*t)
# xの残りの部分を0埋め
#x = np.append(x, np.zeros(L - RT))

# sample: wav
#fs, x = wio.read("shining.wav")
#x = x[:, 0]
fs, x = wio.read("MSK.20100405.M.CS05.wav")
#x = x[:, 0]
x = x[START:END]
print(x)
print("Length: " + str(len(x)))
print("fs: " + str(fs))
if len(x) > L:
    x = x[0:L]
elif len(x) < L:
    x = np.append(x, np.zeros(L - len(x)))

#pxx, freq, bins, t = plt.specgram(x,Fs = fs)
#plt.show()

# 解析
X = np.zeros((M, N), dtype=complex)
future_list = []
start = time.time()
prev_m1 = 0
max_m = M
with ThreadPoolExecutor(max_workers=8) as e:
    for i in range(THREADS):
        m0 = (int)(i * (max_m / THREADS))
        if i == THREADS - 1:
            m1 = max_m
        else:
            m1 = max(prev_m1 + 1, (int)((i + 1) * (max_m / THREADS)))
        prev_m1 = m1
        n0 = 0
        n1 = N
        print("m0:" + str(m0) + ", m1:" + str(m1) + ", n0:" + str(n0) + ", n1:" + str(n1))
        future = e.submit(calc_X, x, w, m0, m1, n0, n1)
        future_list.append(future)

    i = 0
    for future in futures.as_completed(fs=future_list):
        temp_X, m0, m1, n0, n1 = future.result()
        print("complete: " + str(m0) + ":" + str(m1) + ", " + str(n0) + ":" + str(n1))
        X[m0:m1, n0:n1] = temp_X[m0:m1, n0:n1]
        i += 1
print ("time: " + str(time.time()-start))

# 逆変換
start = time.time()
cL = L
cx = np.zeros(cL, dtype=complex)
future_list = []
with ThreadPoolExecutor(max_workers=8) as e:
    for i in range(THREADS):
        l0 = (int)(i * (cL / THREADS))
        if i == THREADS - 1:
            l1 = cL
        else:
            l1 = (int)((i + 1) * (cL / THREADS))
        print("l0:" + str(l0) + ", l1:" + str(l1))
        future = e.submit(calc_cx, X, cw, l0, l1)
        future_list.append(future)

    i = 0
    for future in futures.as_completed(fs=future_list):
        temp_cx, l0, l1 = future.result()
        print("complete: " + str(i))
        cx[l0:l1] = temp_cx[l0:l1]
        i += 1
print ("time: " + str(time.time()-start))
print(cx)

# 逆変換結果をwavファイルに出力
wio.write("output.wav", fs, np.real(cx) / L)

# spectrogram
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()

# プロット用パラメータ
x_max = L
y_max = L / 2
X = X[0:(int)(y_max/b), 0:(int)(x_max/a)]
print(len(X))
print(np.linspace(0, y_max, N))

# decibelize
X2 = db(X, 2e-5)

# 波形
ax1.plot(t, x)

# 1秒分の波形
ax2.stem(t, x, '*')
ax2.set_xlim(0, 100)

# 窓
ax3.plot(w)

# 解析結果
c = ax5.contourf(np.linspace(0, x_max, (int)(x_max/a)), np.linspace(0, y_max, (int)(y_max/b)), np.abs(X2), 50, cmap='jet')
c = ax4.contourf(np.linspace(0, x_max, (int)(x_max/a)), np.linspace(0, y_max, (int)(y_max/b)), np.abs(X), 20, locator=ticker.LogLocator(), cmap='jet')
fig4.colorbar(c)

# 逆変換結果
ax6.plot(t[0:cL], cx)

plt.show()
