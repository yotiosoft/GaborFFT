def DGT(x, w, L, a, b, m, n):
    X = 0
    for l in range(L):
        X += x[l] * w[l - a * n] * np.exp((-2 * np.pi * 1j * b * m * l) / L)
    return X

def hammig_w(t):
    return 0.54 - 0.46 * np.cos((2 * np.pi * t) / T)

def hamming(T):
    W = np.zeros(T)
    for t in range(T):
        W[t] = 0.54 - 0.46 * np.cos((2 * np.pi * t) / T)
    return W
