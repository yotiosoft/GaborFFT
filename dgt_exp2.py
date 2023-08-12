import dgt as mydgt
import scipy.io.wavfile as wio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

# sample: wav
w = mydgt.hammig_w(500)

a = 150
b = 150
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

plt.plot(t, tw)
plt.show()

# 窓関数wを時間微分
dw = np.diff(w)
print(dw)

# 音声を順変換
dgt1 = mydgt.DGT(len(w), a, b, L)
X = dgt1.dgt(x, w)
dgt2 = mydgt.DGT(len(tw), a, b, L)
X_tw = dgt2.dgt(x, tw)
dgt3 = mydgt.DGT(len(dw), a, b, L)
X_dwdt = dgt3.dgt(x, dw)

# 再割当て法
new_X = np.zeros((N*5, M*5), dtype=np.complex)
for n in range(M):
    for m in range(N):
        n_new = n + (np.abs((X_tw[m, n] / X[m, n]).real) / a)
        m_new = m - (np.abs((X_dwdt[m, n] / X[m, n]).imag) / b)
        print(n_new, m_new)
        new_X[int(m_new), int(n_new)] += X[m, n]


# 合成した上で逆変換
#idgt = mydgt.IDGT(len(w), a, b, L)
#cx = idgt.idgt(X, w)

# 逆変換結果をwavファイルに出力
#wio.write("exp2.wav", fs, np.real(cx) * pow(10, -4))

mydgt.plot(x, new_X, None, dw, a, b, N, L, fs)
