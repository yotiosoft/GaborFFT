def gabor(x, w, L, m, n):
    X = 0
    for l in range(L):
        X += x[l] * w[l - n] * np.exp((-2 * np.pi * 1j * m * l) / L)
    return X


