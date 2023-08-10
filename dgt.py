import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import matplotlib.ticker as ticker

import time

THREADS = 8

class DGT:
    def __init__(self, t, a, b, start, end):
        self.T = t
        self.a = a
        self.b = b
        self.START = start
        self.END = end
        self.L = end - start
        self.M = int(self.L / self.b)  # y
        self.N = int(self.L / self.a)  # x

    def calc_DGT(self, x, w, m, n):
        dgt_X = 0
        for l in range(self.a * n, self.a * n + self.T):
            if l >= self.L:
                break
            dgt_X += x[l] * w[l - self.a * n] * np.exp((-2 * np.pi * 1j * m * l) / complex(self.M))
        return dgt_X
    
    def calc_IDGT(self, X, g, w, l):
        idgt_x = 0
        #print("l:" + str(l) + ", g[l][l]:" + str(g[l][l]))
        for n in range(self.N):
            if l - self.a * n < 0 or l - self.a * n >= self.T:
                continue
            for m in range(self.M):
                idgt_x += X[m, n] * g[l - self.a * n] * np.exp((2 * np.pi * 1j * m * l) / complex(self.M))
        
        return idgt_x
    
    def hammig_w(self):
        w = np.zeros(self.T)
        for t in range(self.T):
            w[t] = 0.54 - 0.46 * np.cos((2 * np.pi * t) / self.T)

        return w
    
    def calc_X(self, x, w, m0, m1, n0, n1):
        temp_X = np.zeros((self.M, self.N), dtype=complex)
        for m in range(m0, m1):
            for n in range(n0, n1):
                temp_X[m, n] = self.calc_DGT(x, w, m, n)
                # print("m:" + str(m) + ", n:" + str(n) + " : " + str(temp_X[m, n]))
        return (temp_X, m0, m1, n0, n1)
    
    def calc_cx(self, X, cw, w, l0, l1):
        temp_cx = np.zeros(self.L, dtype=complex)
        for l in range(l0, l1):
            temp_cx[l] = self.calc_IDGT(X, cw, w, l)
        return (temp_cx, l0, l1)
    
    def hammig_cw(self, w):
        sum = np.zeros(self.L)
        for l in range(self.L):
            sum[l] = 0
            for n in range(self.N):
                if l == 10 and n < 10:
                    nw = np.zeros(self.L)
                    nw[l+self.a*n:l+self.a*n+self.T] = w
                    plt.plot(nw)
                sum[l] += np.abs(w[(l + self.a * n) % self.T]) ** 2
        plt.xlim(0, 1000)
        plt.show()

        plt.plot(sum)
        plt.show()

        plt.cla()

        cw_a = np.zeros((self.a, self.a), dtype=complex)
        for l in range(self.a):
            for n in range(self.N):
                cw_a[l][l] += np.abs(w[(l + self.a * n) % self.T]) ** 2
            cw_a[l][l] *= self.M
            cw_a[l][l] = 1 / (np.dot(cw_a[l][l].T, cw_a[l][l]))

        cw = np.zeros((self.L, self.L), dtype=complex)
        for l in range(self.L):
            cw[l][l] = cw_a[l % self.a][l % self.a]

        g = np.zeros(self.T)
        for t in range(self.T):
            g[t] = cw[t][t] * w[t]

        plt.plot(g)
        plt.show()

        return g
    
    def db(self, x, dBref):
        y = 20 * np.log10(x / dBref)                   # 変換式
        return y                                       # dB値を返す
    
    def dgt(self):
        w = self.hammig_w()
        cw = self.hammig_cw(w)

        t = np.linspace(0,self. L, self.L)

        fs, x = wio.read("MSK.20100405.M.CS05.wav")
        x = x[self.START:self.END]
        print(x)
        print("Length: " + str(len(x)))
        print("fs: " + str(fs))
        if len(x) > self.L:
            x = x[0:self.L]
        elif len(x) < self.L:
            x = np.append(x, np.zeros(self.L - len(x)))

        # 解析
        X = np.zeros((self.M, self.N), dtype=complex)
        future_list = []
        start = time.time()
        prev_m1 = 0
        max_m = self.M
        with ThreadPoolExecutor(max_workers=8) as e:
            for i in range(THREADS):
                m0 = (int)(i * (max_m / THREADS))
                if i == THREADS - 1:
                    m1 = max_m
                else:
                    m1 = max(prev_m1 + 1, (int)((i + 1) * (max_m / THREADS)))
                prev_m1 = m1
                n0 = 0
                n1 = self.N
                print("m0:" + str(m0) + ", m1:" + str(m1) + ", n0:" + str(n0) + ", n1:" + str(n1))
                future = e.submit(self.calc_X, x, w, m0, m1, n0, n1)
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
                future = e.submit(self.calc_cx, X, cw, w, l0, l1)
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

        # 逆変換結果をwavファイルに出力
        wio.write("output.wav", fs, np.real(cx))

        # spectrogram
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()

        # プロット用パラメータ
        x_max = self.L
        y_max = self.L / 2
        X = X[0:(int)(y_max/self.b), 0:(int)(x_max/self.a)]
        print(len(X))
        print(np.linspace(0, y_max, self.N))

        # decibelize
        X2 = self.db(X, 2e-5)

        # 波形
        ax1.plot(t, x)

        # 1秒分の波形
        ax2.stem(t, x, '*')
        ax2.set_xlim(0, 100)

        # 窓
        ax3.plot(w)

        # 解析結果
        c = ax5.contourf(np.linspace(0, x_max, (int)(x_max/self.a)), np.linspace(0, y_max, (int)(y_max/self.b)), np.abs(X2), 50, cmap='jet')
        c = ax4.contourf(np.linspace(0, x_max, (int)(x_max/self.a)), np.linspace(0, y_max, (int)(y_max/self.b)), np.abs(X), 20, locator=ticker.LogLocator(), cmap='jet')
        fig4.colorbar(c)

        # 逆変換結果
        ax6.plot(t, cx)

        plt.show()

dgt = DGT(500, 50, 50, 5000, 20000)
dgt.dgt()
