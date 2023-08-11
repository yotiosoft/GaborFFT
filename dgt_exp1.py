import dgt as mydgt
import scipy.io.wavfile as wio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# sample: wav
w = mydgt.hammig_w(500)

a = 50
b = 50
start = 5000
end = 20000
x1, fs1 = mydgt.load_wav("F1AES2_parts.wav", start, end)
L1 = len(x1)
N1 = int(L1 / a)
M1 = int(L1 / b)
x2, fs2 = mydgt.load_wav("M1GIM01.wav", start, end)
L2 = len(x1)
N2 = int(L2 / a)
M2 = int(L2 / b)

if fs1 != fs2:
    print("Error: fs1 != fs2")
    exit(1)

# 順変換
dgt1 = mydgt.DGT(len(w), a, b, L1)
X1 = dgt1.dgt(x1, w)
dgt2 = mydgt.DGT(len(w), a, b, L2)
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

# 逆変換
idgt = mydgt.IDGT(len(w), a, b, L)
cx = idgt.idgt(X, w)

# 逆変換結果をwavファイルに出力
wio.write("output.wav", fs1, np.real(cx) * pow(10, -4))

# matplotlib でスペクトログラムを描画（比較用）
plt.specgram(np.real(cx), Fs = fs1)

# プロット
mydgt.plot(x1, X, cx, a, b, N1, L)
