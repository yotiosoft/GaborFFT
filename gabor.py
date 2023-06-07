import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wio

T = 100
a = 10000
b = 10000
L = 100000
N = int(L / a)
M = int(L / b)

def DGT(x, w, a, b, m, n):
    dgt_X = 0
    for l in range(L):
        dgt_X += x[l] * w[(l - a * n) % T] * np.exp((-2 * np.pi * 1j * b * m * l) / L)
    return dgt_X

def hammig_w(t):
    return 0.54 - 0.46 * np.cos((2 * np.pi * t) / T)

w = np.zeros(T)
for t in range(T):
    w[t] = hammig_w(t)

# sample: sin
#x = np.zeros(L)
#for l in range(L):
#    x[l] = np.sin(np.pi * l)

# sample: wav
fs, x = wio.read("BabyElephantWalk60.wav")
x = x[:L]
L = len(x)
N = int(L / a)
M = int(L / b)
#pxx, freq, bins, t = plt.specgram(x,Fs = fs)
#plt.show()

X = np.zeros((M, N), dtype=complex)
for m in range(M):
    for n in range(N):
        X[m, n] = DGT(x, w, a, b, m, n)
        print("m:" + str(m) + ", n:" + str(n) + " : " + str(X[m, n]))

# time domain
# plt.plot(np.abs(X[:, T - 1]))
# plt.show()

# spectrogram
fig, ax = plt.subplots()
ax.set_yscale('log')

c = ax.contourf(np.abs(X), 20, cmap="jet")
fig.colorbar(c)

plt.show()
