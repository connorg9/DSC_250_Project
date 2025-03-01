import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate
import scipy

def SA(t):
    #unitary gate
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    #pauli x and y operators
    x = np.matrix([[0, 1], [1, 0]])
    y = np.matrix([[0, -1j], [1j, 0]])

    # Time evolution operator: e^(i*t*H/2) where H is the interaction Hamiltonian (X⊗Y - Y⊗X)
    A = scipy.linalg.expm((1j / 2) * t * (np.kron(x, y) - np.kron(y, x)))

    #combine swap and time evolution functions
    SA = swap @ A

    #convert to circuit gate to be used later
    SA = UnitaryGate(SA)

    return SA


def ansatz(t_list):
    q = QuantumRegister(4)
    c = ClassicalRegister(4)
    qc = QuantumCircuit(q, c)

    qc.x(0)
    qc.x(1)

    #apply newly created SA gate to qubits, entangling them
    qc.append(SA(t_list[2]), [1, 2])
    qc.append(SA(t_list[0]), [0, 1])
    qc.append(SA(t_list[3]), [2, 3])
    qc.append(SA(t_list[1]), [1, 2])

    return qc


def r_xxxx(qc):

    qc.h(range(4))
    return qc


def r_yyyy(qc):
    #applies S(dagger) and h to each qubit
    qc.sdg(range(4))
    qc.h(range(4))
    return qc


def e1(counts, shots, p):
    #Calculates expectation value (essentially an average of the counts that are in the 1 state)
    exp_val = 0
    counts = dict(counts)

    for key in list(counts.keys()):
        if key[p] == "1":
            exp_val += counts[key] / shots

    return exp_val


def sign(result, state):
    #determine the sign for later functions
    result_list = [int(x) for x in result]

    #Hamming weight - distance from the intended result
    hw = 0

    for i in range(len(state)):
        if state[i] == 1:
            hw += result_list[i]

    return (-1) ** hw


def e2(counts, shots, state):
    #Calculates expectation value for two qubit pauli operators
    val = 0
    counts = dict(counts)

    for result in list(counts.keys()):
        val += sign(result, state) * counts[result] / shots

    return val


def second_state(i, j):
    # Creates a binary vector with 1s at positions i and j
    # Used to represent which qubits to consider for two-qubit operators
    state = np.zeros(4)
    state[i] = 1
    state[j] = 1

    return state


# %%
def vqe_counts(t_list, pauli, shots):
    #Run circuit
    circ = ansatz(t_list)

    #determine measurement basis depending on input
    if pauli == "I":
        pass
    elif pauli == "X":
        circ = r_xxxx(circ)
    elif pauli == "Y":
        circ = r_yyyy(circ)

    #measure the circuits, swap to specific order
    circ.measure(circ.qubits[0], circ.clbits[1])
    circ.measure(circ.qubits[1], circ.clbits[0])
    circ.measure(circ.qubits[2], circ.clbits[3])
    circ.measure(circ.qubits[3], circ.clbits[2])

    #simulate the circuit
    sim = AerSimulator()
    counts = sim.run(circ, shots=shots).result().get_counts()

    return counts


# %%
def vqe_expect(t_list, constants):
    #Calculates expectation value of hamiltonian
    g = constants[0] #strength of interaction
    shots = constants[1]

    #Use previous function to get counts for each measurement basis
    counts_i = vqe_counts(t_list, "I", shots)
    counts_x = vqe_counts(t_list, "X", shots)
    counts_y = vqe_counts(t_list, "Y", shots)

    E = 0

    #single qubit term contribution
    for p in range(4):
        E += (2 * p - (g / 2)) * e1(counts_i, shots, p)

    #two-qubit term contribution
    for i in range(4):
        for j in range(i + 1, 4):
            state = second_state(i, j)
            E += (-g / 4) * (e2(counts_x, shots, state) + e2(counts_y, shots, state))

    return E


# %%
def pairing_4p4h(g, d):
    #Pairing hamiltonian
    #g = interaction strength, d = particle distance (energy spacing)
    return np.array([
        [2 * d - g, -g / 2, -g / 2, -g / 2, -g / 2, 0],
        [-g / 2, 4 * d - g, -g / 2, -g / 2, 0, -g / 2],
        [-g / 2, -g / 2, 6 * d - g, 0, -g / 2, -g / 2],
        [-g / 2, -g / 2, 0, 6 * d - g, -g / 2, -g / 2],
        [-g / 2, 0, -g / 2, -g / 2, 8 * d - g, -g / 2],
        [0, -g / 2, -g / 2, -g / 2, -g / 2, 10 * d - g]
    ])


# %%
def vqe(t_list, g, shots):
    #define constants used in other functions
    constants = [g, shots]

    #define hamiltonian
    H = pairing_4p4h(g, 1.0)

    #get eigenvalues and eigenvectors from hamiltonian
    e_vals, e_vecs = np.linalg.eig(H)

    #get exact ground state for later comparison
    e_exact = np.real(min(e_vals))

    #optimize parameters
    opt = scipy.optimize.minimize(vqe_expect, x0=t_list, args=constants)

    t_min = opt.x

    #calculate final energy using optimized parameters
    e_est = vqe_expect(t_min, constants)

    return e_est, e_exact, t_min


# %%
import matplotlib.pyplot as plt

g_min = 0
g_max = 1.0
shots = 2 ** 10
d = 1.0

e_ext_list = []

#calculate exact energies for a range of g values (again, interaction strength)
for g in np.linspace(g_min, g_max, 100):
    H = pairing_4p4h(g, d)
    e_vals, e_vecs = np.linalg.eig(H)
    e_ext_list.append(np.real(min(e_vals)))

e_est_list = []

for g in np.linspace(g_min, g_max, 5):
    t_list = (-g / 4) * np.array(
        [1 / (0 - 2 - (g / 2)), 1 / (0 - 3 - (g / 2)), 1 / (1 - 2 - (g / 2)), 1 / (1 - 3 - (g / 2))])

    #run VQE
    e_est, e_exact, t_min = vqe(t_list, g, shots)

    e_est_list.append(e_est)

plt.plot(np.linspace(g_min, g_max, 100), e_ext_list)
plt.scatter(np.linspace(g_min, g_max, 5), e_est_list)
plt.show()