import numpy as np
import matplotlib.pyplot as plt


def lowerb(a, V):
    return np.sum(-np.exp(0.5 * a * a * V.diagonal())) + np.linalg.slogdet(V)[1]


maxiter = 50
inneriter = 5
T = 10
rate = np.empty(T)
sigma = np.eye(T) * 2
omega = np.linalg.inv(sigma)
K = np.eye(T)
V = np.eye(T)
a = 1
epsilon = 1e-5
rate[:] = np.exp(0.5 * a * a * V.diagonal())

old_V = V.copy()

lbound = np.full(maxiter, np.NINF)
lbound[0] = lowerb(a, V)

it = 1
convergent = False
while not convergent and it < maxiter:
    for t in range(T):
        k_ = K[t, t] - 1 / V[t, t]  # \tilde{k}_tt
        old_vtt = V[t, t]
        # fixed point iterations
        for _ in range(inneriter):
            vtt = 1 / (omega[t, t] - k_ + rate[t] * a * a)
            V[t, t] = vtt
            # update rate
            rate[t] = np.exp(0.5 * a * a * V[t, t])
            # update V
            not_t = np.arange(T) != t
            V[np.ix_(not_t, not_t)] = np.nan_to_num(V[np.ix_(not_t, not_t)] +
                                                         (V[t, t] - old_vtt) *
                                                         np.outer(V[t, not_t], V[t, not_t]) /
                                                         (old_vtt * old_vtt))
            V[t, not_t] = V[not_t, t] = V[t, t] * V[t, not_t] / old_vtt
            # update k_tt
            K[t, t] = k_ + 1 / V[t, t]

    lbound[it] = lowerb(a, V)

    if np.linalg.norm(old_V - V) < epsilon:
        convergent = True

    it += 1
    old_V[:] = V

print lbound
print V
plt.plot(lbound)
plt.show()

