from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.qasm3 import dumps
import numpy as np
from collections import defaultdict
import itertools

# Used for decomposing Matrix into Gates
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CZGate

bases = ["1111", "1100", "1010", "0110"]

PC_patterns_16 = []
# builds 16 bit patterns from the bases patterns
for p in bases:
    for q in bases:
        base = ""
        for bit in p:
            if bit == '1':
                base = base + q
            else:
                not_q = ''.join('1' if b == '0' else '0' for b in q)
                base = base + not_q
                
        PC_patterns_16.append(base)

# Reverses bit string
def pattern_to_vector(p):
    n = len(p)
    return [int(p[n - 1 - k]) for k in range(n)]

perfectly_classifiable = [pattern_to_vector(p) for p in PC_patterns_16]

# Constructs the C2 Gate: (H x H) CZ (Z x Z) (H x H)
def build_C2():
    qc = QuantumCircuit(2, name='C2')
    qc.h([0, 1])
    qc.cz(0, 1)
    qc.z([0, 1])
    qc.h([0, 1])
    return qc

def build_C2_prime():

    bits = []
    for base in bases:
        int_bits = [-1 if b == '1' else 1 for b in base]
        bits.append(int_bits)
    
    c2_prime = 0.5 * np.array(bits, dtype=float)
    qc = QuantumCircuit(2, name='C2_prime')

    qc.unitary(c2_prime, [0,1])

    U = Operator(qc).data
    print(c2_prime)

    decomp = transpile(qc, basis_gates=['u', 'h', 'cz', 'z'], optimization_level=0) # quantum circuit

    print(decomp)
    print(dumps(decomp)) # step-by-step instructions

    m = Operator(c2_prime)
    decomposer = TwoQubitBasisDecomposer(CZGate())

    qc = decomposer(m)
    print(qc)
    
    return qc

# Builds an oracle that uses 4 bits for the input and 1 qubit for the label
def build_oracle_4q(pattern_vector):
    qc = QuantumCircuit(5, name='Oracle')

    # Applies controlled X gate where the label is 1
    for idx, label in enumerate(pattern_vector):
        
        if label == 1:
            bits = [(idx >> i) & 1 for i in range(4)]

            # Flips the qubits that are in the 0 state
            for q, bit_value in enumerate(bits):
                if bit_value == 0:
                    qc.x(q)

            qc.mcx([0, 1, 2, 3], 4)

            # Restores the flipped qubits
            for q, bit_value in enumerate(bits):
                if bit_value == 0:
                    qc.x(q)
    return qc

# Builds the C2 x C2 circuit
def run_C2C2(pattern_vector):
    qc = QuantumCircuit(5)
    qc.x(4)
    qc.h(4)
    qc.h([0, 1, 2, 3])
    qc.append(build_oracle_4q(pattern_vector).to_gate(), range(5))
    qc.append(build_C2_prime().to_gate(), [0, 1])   # C2 on qubits 0,1
    qc.append(build_C2_prime().to_gate(), [2, 3])   # C2 on qubits 2,3
    return qc


# Gets the probabilites for the input bases
def get_input_probs_4q(pattern_vector):
    qc = run_C2C2(pattern_vector) # C2C2
    
    sv = Statevector.from_instruction(qc)
    
    probs_full = sv.probabilities() # gets probabilities from statevector
    
    probs = np.zeros(16)

    # Uses the last 4 bits to calculate the corresponding probability
    for i in range(32):
        input_index = i & 0b1111
        probs[input_index] += probs_full[i]
    return probs

# Hamming distance between two patterns
def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


# Calculates the nearest neighbors at the minimum hamming distance
# Returns the nearest neighbors and the minimum distances
def nearest_neighbors(patterns):
    
    distances = [hamming(patterns, perf) for perf in perfectly_classifiable]
    min_dist = min(distances)
    
    # Finds which functions are at the minimum distance
    nearest_neighbor = [i for i, dist in enumerate(distances) if dist == min_dist]
    
    return nearest_neighbor, min_dist

# Checks whether each of the bases is correctly classified
def debug_basis_classification():
    print("DEBUG: C2 x C2 basis pattern classification\n")
    all_correct = True
    for idx, pattern in enumerate(perfectly_classifiable):
        probs = get_input_probs_4q(pattern)
        found = int(np.argmax(probs))
        
        print(f"Basis pattern {idx:>2}: argmax={found:>2}, expected={idx:>2}",
              "Correct" if found == idx else "Incorrect")

# This builds the C2 x C2 classifier,
# and computes the average probability at each hamming distance for all 2^16 patterns
def run_experiment():
    print("Checking all 2^16 functions\n")
    prob_sum = defaultdict(float)
    counts   = defaultdict(int)

    total = 2**16
    for pattern in range(total):
        if pattern % 5000 == 0:
            print(f"  Progress: {pattern}/{total} ({100*pattern/total:.1f}%)")

        # Calculates the pattern bit vector
        vector = [(pattern >> i) & 1 for i in range(16)]
        
        nearest_indices, min_dist = nearest_neighbors(vector) # nearest neighbors
        probs = get_input_probs_4q(vector)

        neighbor_prob = sum(probs[i] for i in nearest_indices)
        prob_sum[min_dist] += neighbor_prob
        counts[min_dist] += 1

    print("\nResults for F_Q2★F_Q2 (Table 8):")
    print(f"{'Hamming Dist':>13} | {'Probability':>10} | {'Count':>8}")
    print("-" * 40)
    for d in range(0, 17):
        if counts[d] > 0:
            threshold = prob_sum[d] / counts[d]
            print(f"{d:>13} | {threshold:>10.4f} | {counts[d]:>8}")
        else:
            print(f"{d:>13} | {'—':>10} | {'—':>8}")

if __name__ == "__main__":
    debug_basis_classification()
    run_experiment()
