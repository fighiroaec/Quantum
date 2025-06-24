from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PhaseGate
import numpy as np

# Prints the state
def print_state_info(qc, title):
    qc.remove_final_measurements()
    state = Statevector.from_instruction(qc)
    print(f"\n{title}\n")
    print("Basis | Amplitude           | Probability | Phase")
    print("------------------------------------------------------")
    # Prints the basis, amplitude, probability and phase
    for i, amp in enumerate(state.data):
        prob = np.abs(amp)**2
        if prob > 1e-6:
            phase = np.angle(amp)
            bitstring = f"{i:0{state.num_qubits}b}"
            reversed_bitstring = bitstring[::-1]  # Bitstring in Little Endian
            print(f"|{reversed_bitstring}⟩ | {amp.real:+.4f}{amp.imag:+.4f}𝑖 | {prob*100:9.4f}% | {phase:+.4f}")


# Little Endian Representation of Integers
def apply_pauli_from_int(qc, value, qubits):
    bin_string = f"{value:0{len(qubits)}b}"
    for idx, bit in enumerate(reversed(bin_string)):
        if bit == '1':
            qc.x(qubits[idx])


def find_angle(j):
    return -2 * np.arccos(np.sqrt((j - 1) / j))

def controlled_si(qc, p, u, i, len_ps):
    angle = find_angle(len_ps + 1 - i)
    qc.cry(angle, u[0], u[1])

def storage_algorithm(qc, ps, p, u, m):
    for idx, val in enumerate(ps):
        
        # Step 0
        apply_pauli_from_int(qc,val, p)

        # Step 1: 2XOR Gate with p and u[1] as controls and m as the target
        for j in range(len(p)):
            qc.ccx(p[j], u[1], m[j])

        # Step 2: First XOR gate, then NOT gate with p as the control and m as the target
        for j in range(len(p)):
            qc.cx(p[j], m[j])
            qc.x(m[j])

        # Step 3: nXOR Gate with m as the control and u[0] as the target
        qc.mcx(m, u[0])

        # Step 4: Controlled S^i Gate
        controlled_si(qc, p, u, idx + 1, len(ps))
        
        # Step 5: nXOR Gate with m as the control and u[0] as the target
        qc.mcx(m, u[0])

        # Step 6: First NOT gate, then NOT gate with p as the control and m as the target
        for j in reversed(range(len(p))):
            qc.x(m[j])
            qc.cx(p[j], m[j])

        # Step 7: 2XOR gate with p and u[1] as controls and m as the gate
        for j in reversed(range(len(p))):
            qc.ccx(p[j], u[1], m[j])

        # Step 8: Clear the pattern register
        apply_pauli_from_int(qc,val, p)

def u_gate(qc, qubit, n):
    qc.x(qubit)
    qc.p(np.pi / (2.0 * n), qubit)
    qc.x(qubit)


def cu_squared(qc, control, target, n):
    # Define CU^-2
    u_adj = PhaseGate(-np.pi / (n)).control(1)

    # Apply U^-2 between X Gates
    qc.cx(control, target)
    qc.append(u_adj, [control, target])
    qc.cx(control, target)

def apply_u_and_cu(qc, m, c):
    n = len(m)
    
    # Apply U
    for q in m:
        u_gate(qc, q, n)
    #print_state_info(qc, "U Gate")
    
    # Apply CU^-2
    for q in m:
        cu_squared(qc, c, q, n)
    #print_state_info(qc, "CU Gate")

def retrieval_algorithm(qc, i, m, c):
    # Step 0: H gate for c qubit
    qc.h(c)
    #print_state_info(qc, "Hadamard Gate")
    
    # Step 1: First XOR gate, then NOT gate with m as the target and i as the control
    for j in range(len(i)):
        qc.cx(i[j], m[j])
        qc.x(m[j])
    #print_state_info(qc, "XOR and NOT Gates")
    
    # Step 2: Unitary Exponential Operator
    apply_u_and_cu(qc, m, c)
    
    # Step 3: First NOT gate, then XOR gate with m as the target and i as the control
    for j in reversed(range(len(i))):
        qc.x(m[j])
        qc.cx(i[j], m[j])
    #print_state_info(qc, "NOT and XOR Gates")
    
    
    qc.h(c)

def bv_algorithm(qc, i, m, c, degree_threshold):
    for q in i:
        qc.h(q)
    retrieval_algorithm(qc, i, m, c)
    for q in i:
        qc.h(q)

def run_quantum_memory():

    # Registers
    i = QuantumRegister(3, 'i')  # Input Pattern Register
    u = QuantumRegister(2, 'u')  # Intermediate Register
    m = QuantumRegister(3, 'm')  # Memory Register
    qc = QuantumCircuit(i,u,m)

    qc.x(u[1])
    print_state_info(qc, "Initial State")

    
    ps = [1,4] # Patterns
    #print(f"Storing patterns: {ps}")
    
    storage_algorithm(qc, ps, i, u, m)
    
    print_state_info(qc, "Storage Results")
    
    #qc.x(i[0])
    qc.x(i[2])

    retrieval_algorithm(qc,i,m,u[0])
    print_state_info(qc, "Retrieval Results")

    print(f"\nControl and Label Qubits:\n")

    # Prints the probabilities for the control and label qubits
    state = Statevector.from_instruction(qc)
    probabilities = state.probabilities(qargs=[5, 3])
    for i, p in enumerate(probabilities):
        print(f"{i:02b}: {p:.4f}")



# Execute the full program
run_quantum_memory()
