import mydgt
import scipy.io.wavfile as wio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

# sample: wav
w = mydgt.hammig_w(200)
plt.plot(w)
plt.show()

a = 50
b = 50
start = 0
end = sys.maxsize    # ファイル全体

# 音声を読み込み
x, fs = mydgt.load_wav("MSK.20100405.M.CS05.wav", start, end)
L = len(x)

N = int(L / a)
M = int(L / b)

# 窓関数wに時間変数tで重み付け
t = np.arange(-len(w)/2, len(w)/2)
tw = w * t
plt.plot(tw)
plt.show()

# 窓関数wを時間微分
dw = np.diff(w)
print(dw)
plt.plot(dw)
plt.show()

# 音声を順変換
dgt1 = mydgt.DGT(len(w), a, b, L)
X = dgt1.dgt(x, w)
dgt2 = mydgt.DGT(len(tw), a, b, L)
X_tw = dgt2.dgt(x, tw)
dgt3 = mydgt.DGT(len(dw), a, b, L)
X_dwdt = dgt3.dgt(x, dw)

# 再割当て法
new_X = np.zeros((M, N), dtype=np.complex)
for m in range(M):
    for n in range(N):
        n_new = n + (np.abs((X_tw[m, n] / X[m, n]).real))
        m_new = m - (np.abs((X_dwdt[m, n] / X[m, n]).imag))
        print(n_new, m_new)
        if int(n_new) >= N or int(m_new) >= M:
            continue
        new_X[int(m_new), int(n_new)] += X[m, n]

# スパース化後のスペクトログラムを描画
mydgt.plot(x, new_X, None, dw, a, b, N, L, fs)
