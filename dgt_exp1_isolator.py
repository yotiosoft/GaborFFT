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

# 合成音声を読み込み
x_mixed, fs = mydgt.load_wav("mixed_wave.wav", start, end)
x1 = np.append(x_mixed, np.zeros(b - (len(x_mixed) % b)))               # a, b で割り切れる数まで0埋め
L_mixed = len(x_mixed)

# 分離したい音声を読み込み
x_target, fs = mydgt.load_wav("M1GIM01.wav", start, end)
x_target = np.append(x_target, np.zeros(b - (len(x_target) % b)))       # a, b で割り切れる数まで0埋め
L_target = len(x_target)

if L_mixed > L_target:
    x_target = np.append(x_target, np.zeros(L_mixed - L_target))
else:
    x_mixed = np.append(x_mixed, np.zeros(L_target - L_mixed))
L = max(L_mixed, L_target)

N = int(L / a)
M = int(L / b)

# 各音声を順変換
dgt_mixed = mydgt.DGT(len(w), a, b, L)
X_mixed = dgt_mixed.dgt(x_mixed, w)
dgt_target = mydgt.DGT(len(w), a, b, L)
X_target = dgt_target.dgt(x_target, w)

# 分離
X_mixed -= X_target

print(X_mixed)

# 合成した上で逆変換
idgt = mydgt.IDGT(len(w), a, b, L)
cx = idgt.idgt(X_mixed, w)

# 逆変換結果をwavファイルに出力
wio.write("isolated_wave2.wav", fs, np.real(cx) * pow(10, -4))

mydgt.plot(x_mixed, X_mixed, cx, w, a, b, N, L, fs)
