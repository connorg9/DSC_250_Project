
import numpy as np
import matplotlib.pyplot as plt

def schrodinger(n, a):

    psi = []
    E_n = []
    pos = []
    prob = []

    for x in np.linspace(-1, a + 1, 1000):

        if x <= 0 or x >= a:
            #Bad, not allowed (inside walls), infinite potential
            pos.append(x)
            psi.append(0)
            E_n.append(0)
            prob.append(0)

        else:
            #Good, between walls, potential = 0
            print("Good, allowed")
            m = 9.10938356e-31 #kg
            h = 6.62607004e-34 #Joule-seconds
            hbar = h / (2 * np.pi)
            k_n = (n * np.pi) / a

            A = np.sqrt(2 / a)

            pos.append(x)
            psi.append(A * np.sin(k_n * x))
            E_n.append(((hbar ** 2) * (k_n ** 2)) / (2 * m))
            prob.append(psi[-1] ** 2)

    return pos, psi, E_n, prob

a = 1
pos, psi, E_n, prob = schrodinger(1, a)
pos2, psi2, E_n2, prob2 = schrodinger(2, a)

plt.figure()
plt.plot(pos, prob, c = "b", label = "n = 1")
plt.plot(pos2, prob2, c = "r", label = "n = 2")
plt.xlabel("Position")
plt.ylabel("Probability")
plt.legend(loc = "best")
plt.vlines(0, 0, 5, linestyles = "dashed")
plt.vlines(a, 0, 5, linestyles = "dashed")
plt.xlim(-1, a + 1)
plt.ylim(-1, 5)
plt.show()