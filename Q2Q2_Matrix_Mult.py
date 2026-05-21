from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
import numpy as np
from collections import defaultdict

basis1 = ["1111", "1100", "1010", "0110"]
basis2 = basis1

PC_patterns_16 = []
# builds 16 bit patterns from the bases patterns
for p in basis1:
    for q in basis2:
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

def build_B1():
    qc = QuantumCircuit(1, name='B1')
    qc.h(0)
    U = Operator(qc).data
    print(U)
    return qc

def build_C2_prime(basis, n):

    bits = []
    for base in basis:
        int_bits = [-1 if b == '1' else 1 for b in base]
        bits.append(int_bits)
        
    c2_prime = 1/n**(1/2) * np.array(bits)
    
    qc = QuantumCircuit(n//2, name='C2_prime')

    qc.unitary(c2_prime, list(range(0, n//2)))
    
    return qc

# Creates the Tensor Product between C2' and C2'
def build_classifier_matrix():

    c2 = Operator(build_C2_prime(basis1, len(basis1[0]))).data
    b1 = Operator(build_C2_prime(basis2, len(basis2[0]))).data
    classifier = np.kron(c2, b1)  # C2 x B1
    
    return classifier


# Computes probabilities for all 2^16 bit pattern vectors
def compute_all_probs(classifier, n):
    
    num_functions = 2**n
    
    probs = np.zeros((num_functions, n), dtype=float)
    
    for pattern in range(num_functions):
        
        # This shifts the pattern to the ith bit
        pattern_vector = []
        for i in range(n):
            bit = (pattern >> i) & 1
            pattern_vector.append(bit)
        
        # Builds the state vector
        psi_2 = np.zeros(n, dtype=complex)
        for i in range(n):
            phase = 1.0 if pattern_vector[i] == 0 else -1.0
            psi_2[i] = phase * (1.0 / n**(1/2))
        
        # Applys the classifier matrix to the state vector
        psi_3 = np.zeros(n, dtype=complex)
        for i in range(n):
            total = 0
            for j in range(n):
                total += classifier[i, j] * psi_2[j]
            psi_3[i] = total
        
        # Compute measurement probability for each pattern at each perfectly classifiable pattern
        for i in range(n):
            probs[pattern, i] = abs(psi_3[i]) ** 2
    
    return probs

# Computes Minimum Hamming Distance from each of the 65536 bit patterns to one of the pattern bases
# Returns their distances
def hamming_distances_to_pc(all_bits):
    
    num_functions = len(all_bits)      # 2^n
    num_patterns = len(perfectly_classifiable)  # n
    pattern_length = len(perfectly_classifiable[0])
    
    distances = np.zeros((num_functions, num_patterns), dtype=int)

    # Calculates the number of differences between each bit pattern and a perfectly classifiable one
    for pattern in range(num_functions):
        for i in range(num_patterns):
            dist = 0
            for j in range(pattern_length):
                if all_bits[pattern][j] != perfectly_classifiable[i][j]:
                    dist += 1
            distances[pattern][i] = dist
    
    return distances


# This builds the C2 x C2 classifier,
# applies classifier to bit patterns and computes the minimum hamming distances to the bases,
# and prints a table of average probabilities and counts for each hamming distance
def run_experiment():
    
    classifier = build_classifier_matrix() # C2 x C2

    
    all_patterns = []

    num_bits = len(perfectly_classifiable[0])
    
    for pattern in range(2**num_bits):
        pattern_vector = []
        # Extracts bit from pattern
        for i in range(num_bits):
            bit = (pattern >> i) & 1
            pattern_vector.append(bit)
        all_patterns.append(pattern_vector) # Adds pattern to all_patterns
    bits = np.array(all_patterns, dtype=np.float32)

    probs = compute_all_probs(classifier, num_bits) # probability calculations

    print("Computing Hamming distances")
    ham_d = hamming_distances_to_pc(bits)
    min_dist = ham_d.min(axis=1) # minimum hamming distances for each pattern

    
    prob_sum  = defaultdict(float)
    counts    = defaultdict(int)

    # Calculates the probability sums and the counts
    for i in range(2**num_bits):
        distance = int(min_dist[i])
        nearest_indices = np.where(ham_d[i] == distance)[0]
        nearest_prob = probs[i, nearest_indices].sum()
        prob_sum[distance] += float(nearest_prob)
        counts[distance]   += 1

    # Prints table with counts and average probabilities for each hamming distance
    print("\nResults for F_Q2★F_Q2 (Table 8):")
    print(f"{'Hamming Dist':>13} | {'Probability':>10} | {'Count':>8}")
    print("-" * 40)
    for distance in range(num_bits + 1):
        if counts[distance] > 0:
            threshold = prob_sum[distance] / counts[distance]
            print(f"{distance:>13} | {threshold:>10.4f} | {counts[distance]:>8}")
        else:
            print(f"{distance:>13} | {'—':>10} | {'—':>8}")

# Checks whether each of the bases is correctly classified
def debug_basis_classification():
    num_bits = len(perfectly_classifiable[0])

    
    print("DEBUG: C2 x C2 basis pattern classification\n")
    C2C2_classifier = build_classifier_matrix()
    all_x = np.arange(2**num_bits, dtype=np.uint32)
    bits = ((all_x[:, None] >> np.arange(num_bits)[None, :]) & 1).astype(np.float32)
    probs = compute_all_probs(C2C2_classifier, num_bits)

    
    for idx, pattern_vector in enumerate(perfectly_classifiable):
        x = sum(pattern_vector[i] << i for i in range(num_bits))
        p = probs[x]
        found = int(np.argmax(p))
        
        print(f"Basis pattern {idx:>2}: argmax={found:>2}, expected={idx:>2}",
              "Correct" if found == idx else "Incorrect")

if __name__ == "__main__":
    debug_basis_classification()
    run_experiment()
