def DGT(x, w, L, a, b, m, n):
    X = 0
    for l in range(L):
        X += x[l] * w[l - a * n] * np.exp((-2 * np.pi * 1j * b * m * l) / L)
    return X
