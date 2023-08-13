import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import matplotlib.ticker as ticker
import time
import sys

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

def plot(x, X, tx, w, a, b, N, L, fs):
    t = np.linspace(0, L, L)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    # プロット用パラメータ
    x_max = L
    y_max = L / 2
    X = X[0:(int)(y_max/b), 0:(int)(x_max/a)]

    # 元々の時間領域信号
    ax1.plot(t, x)

    # 窓関数
    ax2.plot(w)

    # 順変換結果
    c = ax3.contourf(np.linspace(0, x_max, (int)(x_max/a)), np.linspace(0, int(fs/2), int(y_max/b)), np.abs(X), 20, locator=ticker.LogLocator(), cmap='jet')
    fig3.colorbar(c)

    # 逆変換結果
    if tx is not None:
        ax4.plot(t, tx)

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
            dgt_X += x[l] * w[l - self.a * n] * np.exp((-2 * np.pi * 1j * ((m * l) % self.L)) / complex(self.M))
        return dgt_X
    
    def X_part(self, x, w, m0, m1, n0, n1):
        temp_X = np.zeros((self.M, self.N), dtype=complex)
        for m in range(m0, m1):
            for n in range(n0, n1):
                temp_X[m, n] = self.do_DGT(x, w, m, n)
        return (temp_X, m0, m1, n0, n1)
    
    def dgt(self, x, w):
        # 解析
        start = time.time()
        X = np.zeros((self.M, self.N), dtype=complex)
        for m in range(self.M):
            for n in range(self.N):
                X[m, n] = self.do_DGT(x, w, m, n)
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
    
    def do_IDGT(self, X, g, l):
        idgt_x = 0
        #print("l:" + str(l) + ", g[l][l]:" + str(g[l][l]))
        for n in range(self.N):
            if l - self.a * n < 0 or l - self.a * n >= self.T:
                continue
            for m in range(self.M):
                idgt_x += X[m, n] * g[(l - self.a * n) % self.T] * np.exp((2 * np.pi * 1j * ((m * l) % self.L)) / complex(self.M))
        
        return idgt_x

    def slide_window(self, w, l):
        new_w = np.zeros(self.L)
        for t in range(self.T):
            new_w[(l + t) % self.L] = w[t]
        return new_w
    
    def hammig_g(self, w):
        cw_a = np.zeros(self.L, dtype=complex)
        nw = self.slide_window(w, 0)
        for l in range(self.T):
            sum = 0
            for n in range(self.N):
                sum += np.abs(nw[(l + self.a * n) % self.L]) ** 2
            cw_a[l] = self.M * sum
        cw_a = 1 / cw_a

        g = np.zeros(self.T, dtype=complex)
        g = cw_a[0:self.T] * w[0:self.T]

        return g
    
    def idgt(self, X, w):
        g = self.hammig_g(w)

        # 逆変換
        start = time.time()
        tx = np.zeros(self.L, dtype=complex)
        for l in range(self.L):
            tx[l] = self.do_IDGT(X, g, l)
        print ("time: " + str(time.time()-start))

        return tx

if __name__ == "__main__":
    # 窓：ハミング窓
    # 窓長L：500
    w = hammig_w(500)
    # スライド幅
    a = 50
    b = 25

    # 音声ファイル読み込み
    start = 0
    end = sys.maxsize    # ファイル全体
    x, fs = load_wav("MSK.20100405.M.CS05.wav", start, end)

    # x を a, b で割り切れる数まで0埋め
    x = np.append(x, np.zeros(a - (len(x) % a)))
    print("len(x):" + str(len(x)))
    L = len(x)
    N = int(L / a)
    M = int(L / b)

    # 順変換
    dgt = DGT(len(w), a, b, L)
    X = dgt.dgt(x, w)
    print(X)

    # 逆変換
    idgt = IDGT(len(w), a, b, L)
    tx = idgt.idgt(X, w)

    # 逆変換結果をwavファイルに出力
    wio.write("output.wav", fs, np.real(tx) * pow(10, -4))

    # matplotlib でスペクトログラムを描画（比較用）
    plt.specgram(np.real(x), Fs = fs)

    #for l in range(L):
    #     print("l:" + str(l) + ", x[l]:" + str(x[l]) + ", tx[l]:" + str(tx[l]))

    # プロット
    plot(x, X, tx, w, a, b, N, L, fs)
