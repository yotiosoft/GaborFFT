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
x, fs = mydgt.load_wav("shining.wav", start, end)
x = x[0:100000]
L = len(x)

N = int(L / a)
M = int(L / b)

# 各音声を順変換
dgt1 = mydgt.DGT(len(w), a, b, L)
X = dgt1.dgt(x, w)

print(X)

# 合成した上で逆変換
idgt = mydgt.IDGT(len(w), a, b, L)
cx = idgt.idgt(X, w)

# 逆変換結果をwavファイルに出力
wio.write("exp2_output.wav", fs, np.real(cx) * pow(10, -4))

mydgt.plot(x, X, cx, w, a, b, N, L, fs)
