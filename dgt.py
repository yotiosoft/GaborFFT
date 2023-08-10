import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import matplotlib.ticker as ticker

import time

THREADS = 8

def load_wav(path, start, end):
    fs, x = wio.read(path)
    print(x)
    print("Length: " + str(len(x)))
    print("fs: " + str(fs))
    if x.ndim == 2:
        x = x[:, 0]
    x = x[start:min(end, len(x))]
    return (x, fs)

def hammig_w(T):
    w = np.zeros(T)
    for t in range(T):
        w[t] = 0.54 - 0.46 * np.cos((2 * np.pi * t) / T)
    return w

def db(x, dBref):
    y = 20 * np.log10(x / dBref)                   # 変換式
    return y                                       # dB値を返す

def plot(x, X, cx, a, b, N, L):
    t = np.linspace(0, L, L)

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
    ax6.plot(t, cx)

    plt.show()

class DGT:
    def __init__(self, t, a, b, l):
        self.T = t
        self.a = a
        self.b = b
        self.L = l
        self.M = int(self.L / self.b)  # y
        self.N = int(self.L / self.a)  # x

    def do_DGT(self, x, w, m, n):
        dgt_X = 0
        for l in range(self.a * n, self.a * n + self.T):
            if l >= self.L:
                break
            dgt_X += x[l] * w[l - self.a * n] * np.exp((-2 * np.pi * 1j * m * l) / complex(self.M))
        return dgt_X
    
    def X_part(self, x, w, m0, m1, n0, n1):
        temp_X = np.zeros((self.M, self.N), dtype=complex)
        for m in range(m0, m1):
            for n in range(n0, n1):
                temp_X[m, n] = self.do_DGT(x, w, m, n)
                # print("m:" + str(m) + ", n:" + str(n) + " : " + str(temp_X[m, n]))
        return (temp_X, m0, m1, n0, n1)
    
    def X_threads(self, x, w, max_m, thread_num = 8):
        X = np.zeros((self.M, self.N), dtype=complex)
        future_list = []
        prev_m1 = 0
        with ThreadPoolExecutor(max_workers=thread_num) as e:
            for i in range(thread_num):
                m0 = (int)(i * (max_m / thread_num))
                if i == thread_num - 1:
                    m1 = max_m
                else:
                    m1 = max(prev_m1 + 1, (int)((i + 1) * (max_m / thread_num)))
                prev_m1 = m1
                n0 = 0
                n1 = self.N
                print("m0:" + str(m0) + ", m1:" + str(m1) + ", n0:" + str(n0) + ", n1:" + str(n1))
                future = e.submit(self.X_part, x, w, m0, m1, n0, n1)
                future_list.append(future)

            i = 0
            for future in futures.as_completed(fs=future_list):
                temp_X, m0, m1, n0, n1 = future.result()
                print("complete: " + str(m0) + ":" + str(m1) + ", " + str(n0) + ":" + str(n1))
                X[m0:m1, n0:n1] = temp_X[m0:m1, n0:n1]
                i += 1

        return X
    
    def dgt(self, x, w):
        # 解析
        start = time.time()
        X = self.X_threads(x, w, self.M)
        print ("time: " + str(time.time()-start))

        return X

class IDGT:
    def __init__(self, t, a, b, l):
        self.T = t
        self.a = a
        self.b = b
        self.L = l
        self.M = int(self.L / self.b)  # y
        self.N = int(self.L / self.a)  # x
    
    def do_IDGT(self, X, g, w, l):
        idgt_x = 0
        #print("l:" + str(l) + ", g[l][l]:" + str(g[l][l]))
        for n in range(self.N):
            if l - self.a * n < 0 or l - self.a * n >= self.T:
                continue
            for m in range(self.M):
                idgt_x += X[m, n] * g[l - self.a * n] * np.exp((2 * np.pi * 1j * m * l) / complex(self.M))
        
        return idgt_x
    
    def hammig_g(self, w):
        sum = np.zeros(self.N)
        for l in [10]:
            for n in range(self.N):
                if l == 10 and n < 10:
                    nw = np.zeros(self.L)
                    nw[l+self.a*n:l+self.a*n+self.T] = w
                    plt.plot(nw)
                sum[n] += np.abs(w[(l + self.a * n) % self.T]) ** 2
        plt.xlim(0, 1000)
        plt.show()

        plt.plot(sum)
        plt.show()

        plt.cla()

        cw_a = np.zeros(self.T, dtype=complex)
        for l in range(self.T):
            sum = 0
            for n in range(self.N):
                sum += np.abs(w[(l + self.a * n) % self.T]) ** 2
            print("{0} {1}".format(l, sum))
            cw_a[l] = self.M * sum
        cw_a = 1 / cw_a

        g = np.zeros(self.T, dtype=complex)
        g = cw_a * w

        plt.plot(cw_a)
        plt.show()
        plt.plot(g)
        plt.show()

        return g
    
    def cx_part(self, X, cw, w, l0, l1):
        temp_cx = np.zeros(self.L, dtype=complex)
        for l in range(l0, l1):
            temp_cx[l] = self.do_IDGT(X, cw, w, l)
        return (temp_cx, l0, l1)
    
    def idgt(self, X, w):
        g = self.hammig_g(w)

        # 逆変換
        start = time.time()
        cx = np.zeros(self.L, dtype=complex)
        future_list = []
        with ThreadPoolExecutor(max_workers=8) as e:
            for i in range(THREADS):
                l0 = (int)(i * (self.L / THREADS))
                if i == THREADS - 1:
                    l1 = self.L
                else:
                    l1 = (int)((i + 1) * (self.L / THREADS))
                print("l0:" + str(l0) + ", l1:" + str(l1))
                future = e.submit(self.cx_part, X, g, w, l0, l1)
                future_list.append(future)

            i = 0
            for future in futures.as_completed(fs=future_list):
                temp_cx, l0, l1 = future.result()
                print("complete: " + str(i))
                cx[l0:l1] = temp_cx[l0:l1]
                i += 1
        print ("time: " + str(time.time()-start))
        print("cx:")
        print(cx)

        return cx

w = hammig_w(500)

a = 250
b = 50
start = 5000
end = 20000
x, fs = load_wav("MSK.20100405.M.CS05.wav", start, end)
L = len(x)
N = int(L / a)

# 順変換
dgt = DGT(len(w), a, b, L)
X = dgt.dgt(x, w)

# 逆変換
idgt = IDGT(len(w), a, b, L)
cx = idgt.idgt(X, w)

# 逆変換結果をwavファイルに出力
wio.write("output.wav", fs, np.real(cx))

# プロット
plot(x, X, cx, a, b, N, L)
