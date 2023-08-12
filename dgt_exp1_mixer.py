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
x1, fs1 = mydgt.load_wav("F1AES2_parts.wav", start, end)
x1 = np.append(x1, np.zeros(b - (len(x1) % b)))               # a, b で割り切れる数まで0埋め
x2, fs2 = mydgt.load_wav("M1GIM01.wav", start, end)
x2 = np.append(x2, np.zeros(b - (len(x2) % b)))               # a, b で割り切れる数まで0埋め
L1 = len(x1)
L2 = len(x2)

if L1 > L2:
    x2 = np.append(x2, np.zeros(L1 - L2))
else:
    x1 = np.append(x1, np.zeros(L2 - L1))
L = max(L1, L2)

N = int(L / a)
M = int(L / b)

if fs1 != fs2:
    print("Error: fs1 != fs2")
    exit(1)
fs = fs1

# 各音声を順変換
dgt1 = mydgt.DGT(len(w), a, b, L)
X1 = dgt1.dgt(x1, w)
dgt2 = mydgt.DGT(len(w), a, b, L)
X2 = dgt2.dgt(x2, w)

# 合成
X = X2.copy()
X[0:M, 0:N] += X1

print(X)

# 合成した上で逆変換
idgt = mydgt.IDGT(len(w), a, b, L)
cx = idgt.idgt(X, w)

# 逆変換結果をwavファイルに出力
wio.write("mixed_wave.wav", fs1, np.real(cx))

mydgt.plot(x1, X1, cx, w, a, b, N, L, fs)
mydgt.plot(x2, X2, cx, w, a, b, N, L, fs)
mydgt.plot(x2, X, cx, w, a, b, N, L, fs)
