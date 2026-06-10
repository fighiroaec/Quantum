import sys

#converts integer to bit pattern
def to_bits(value, n):
    return format(value, f'0{n}b')

# counts number of ones in pattern
def num_ones(pattern):
    return bin(pattern).count("1")


def generate_bases(n):

    
    num_patterns = 2 ** n
    half_ones = 2 ** (n-1)

    

    patterns = list(range(2 ** num_patterns))
    

    # finds all patterns that produce an XOR of half 1's
    xor_four = {}
    for p in patterns:
        xor_four[p] = [q for q in patterns if num_ones(p ^ q) == half_ones]

    bases = []


    # recursively constructs a basis where the XOR of any two patterns has 2^(n-1) 1's and 0's
    def find_basis(found, rest):
        # Took longer at 200000 bases
        if len(bases) >= 200000:
            return

        # Returns if the max number of patterns have been found
        if len(found) == num_patterns:
            bases.append(tuple(found))
            if len(bases) >= 200000:
                return
            if len(bases) % 1000 == 0:
                print(f"{len(bases)}")
            return

        
        for r in rest:
            
            if r <= found[len(found) - 1]:
                continue
            
            # Checks whether all patterns fufill the half 1's XOR condition                
            if all(num_ones(x ^ r) == half_ones for x in found):
                new_rest = [vector for vector in rest
                            if vector > r and num_ones(vector ^ r) == half_ones]
                find_basis(found + [r], new_rest)

    # Only computes bases containing the all zeros pattern
    """start = 0
    start_patterns = xor_four[start]
    find_basis([start], start_patterns)"""

    # Computes all bases, but is large for n>=4
    for start in range(256):
        start_patterns = xor_four[start]
        find_basis([start], start_patterns)

    print(f"\n{len(bases)} bases")
    return bases


def main():
    n = 3

    bases = generate_bases(n)

    output = f"bases_n{n}.txt"
    with open(output, "w") as f:
        for basis in bases:
            for pattern in basis:
                f.write(to_bits(pattern, 2**n) + "\n")
            f.write("\n")

    print(f"Wrote bases to {output}")


if __name__ == "__main__":
    main()
