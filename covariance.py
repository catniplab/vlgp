import numpy as np
import matplotlib.pyplot as plt


def lowerb(a, omega, V):
    return np.sum(-np.exp(0.5 * a * a * V.diagonal())) - 0.5 * np.trace(np.dot(omega, V)) + 0.5 * np.linalg.slogdet(V)[1]


maxiter = 50
inneriter = 5
T = 5
rate = np.empty(T)
sigma = np.eye(T) * 1
omega = np.eye(T) * 1
K = omega.copy()
V = sigma.copy()
a = 1
epsilon = 1e-5
rate[:] = np.exp(0.5 * a * a * V.diagonal())

old_V = V.copy()

lbound = np.full(maxiter, np.NINF)
lbound[0] = lowerb(a, omega, V)

it = 1
convergent = False
while not convergent and it < maxiter:
    for t in range(T):
        k_ = K[t, t] - 1 / V[t, t]
        old_vtt = V[t, t]
        # fixed point iterations
        for _ in range(inneriter):
            V[t, t] = 1 / (omega[t, t] - k_ + np.exp(0.5 * a * a * V[t, t]) * a * a)
            # rate[t] =
            # update V
        # not_t = np.arange(T) != t
        # V[np.ix_(not_t, not_t)] = np.nan_to_num(V[np.ix_(not_t, not_t)] +
        #                                                  (V[t, t] - old_vtt) *
        #                                                  np.outer(V[t, not_t], V[t, not_t]) /
        #                                                  (old_vtt * old_vtt))
        # V[t, not_t] = V[not_t, t] = V[t, t] * V[t, not_t] / old_vtt
        # update k_tt
        K[t, t] = k_ + 1 / V[t, t]

    lbound[it] = lowerb(a, omega, V)

    # if np.linalg.norm(old_V - V) < epsilon:
    #     convergent = True

    it += 1
    old_V[:] = V

print lbound
print V
plt.plot(lbound)
plt.title('Lower bound')
plt.show()

# s = 1
# v = s
# for i in range(50):
#     v = 1 / (s + np.exp(0.5 * v))
#     print v
#     print -np.exp(0.5 * v) - 0.5 * v / s + 0.5 * np.log(v)

