import quimb.tensor as qtn
import numpy as np

def build_portfolio_hamiltonian(mu, Q, top_k=30):

    n = len(mu)

    H = qtn.SpinHam1D(S=0.5)

    # 1. return term (CRITICAL)
    for i in range(n):
        H += -mu[i], "Z", i

    # 2. sparse covariance
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            pairs.append((abs(Q[i, j]), i, j))

    pairs.sort(reverse=True)

    for _, i, j in pairs[:top_k]:
        H[i, j] += Q[i, j], "Z", "Z"

    return H