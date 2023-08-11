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
x2, fs2 = mydgt.load_wav("M1GIM01.wav", start, end)
L1 = len(x1)
L2 = len(x2)

if L1 > L2:
    a1 = a
    b1 = b
    N1 = N2 = int(L1 / a)
    M1 = M2 = int(L1 / b)
    a2 = int(a1 * L2 / L1)
    b2 = int(b1 * L2 / L1)
else:
    a2 = a
    b2 = b
    N1 = N2 = int(L2 / a)
    M1 = M2 = int(L2 / b)
    a1 = int(a2 * L1 / L2)
    b1 = int(b2 * L1 / L2)

if fs1 != fs2:
    print("Error: fs1 != fs2")
    exit(1)

# 各音声を順変換
dgt1 = mydgt.DGT(len(w), a1, b1, L1)
X1 = dgt1.dgt(x1, w)
dgt2 = mydgt.DGT(len(w), a2, b2, L2)
X2 = dgt2.dgt(x2, w)

# 合成
if L1 > L2:
    X = X1.copy()
    X[0:M2, 0:N2] += X2
    L = L1
else:
    X = X2.copy()
    X[0:M1, 0:N1] += X1
    L = L2

print(X)

# 合成した上で逆変換
idgt = mydgt.IDGT(len(w), a, b, L)
cx = idgt.idgt(X, w)

# 逆変換結果をwavファイルに出力
wio.write("mixed_wave.wav", fs1, np.real(cx) * pow(10, -4))

mydgt.plot(x1, X1, cx, w, a1, b1, N1, L1)
mydgt.plot(x2, X2, cx, w, a2, b2, N2, L2)
