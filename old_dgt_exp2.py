import mydgt
import scipy.io.wavfile as wio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

# sample: wav
w = mydgt.hammig_w(500)

a = 200
b = 50
start = 1000000
end = 1050000

# 音声を読み込み
x, fs = mydgt.load_wav("shining.wav", start, end)
L = len(x)

N = int(L / a)
M = int(L / b)

clear_fs_min = 50
clear_m_min = int((M / fs) * clear_fs_min)
clear_fs_max = 8000
clear_m_max = int((M / fs) * clear_fs_max)
for i in range(clear_m_min, clear_m_max, int((M / fs) * 200)):
    print(i)

# 音声を順変換
dgt1 = mydgt.DGT(len(w), a, b, L)
X = dgt1.dgt(x, w)

# ボーカルの音声を除去
# 300Hz〜, 350Hzずつ除去
"""
clear_fs_min = 50
clear_m_min = int((M / fs) * clear_fs_min)
clear_fs_max = 20000
clear_m_max = int((M / fs) * clear_fs_max)
for i in range(clear_m_min, clear_m_max, int((M / fs) * 200)):
    X[i:min(i+2, M/2), :] = 0
    X[max(M-1-i-2, M/2):M-1-i, :] = 0
"""
# 旧版
"""
clear_fs_min = 100
clear_m_min = int(M / fs * clear_fs_min)
clear_fs_max = 1000
clear_m_max = int(M / fs * clear_fs_max)
X[clear_m_min:clear_m_max, :] = 0
X[M-1-clear_m_max:M-1-clear_m_min, :] = 0
"""
print(X)

# 合成した上で逆変換
idgt = mydgt.IDGT(len(w), a, b, L)
tx = idgt.idgt(X, w)

# 逆変換結果をwavファイルに出力
wio.write("exp2_output_only_inst.wav", fs, np.real(tx) * pow(10, -4))

mydgt.plot(x, X, tx, w, a, b, N, L, fs)
