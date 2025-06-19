namespace QuantumMemory {
    import Std.Math.Sqrt;
    import Std.Diagnostics.DumpMachine;
    import Std.Convert.IntAsDouble;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Math;

    // Storage Algorithm
    operation StorageAlgorithm(ps: Int[], p: Qubit[], u: Qubit[], m: Qubit[]) : Unit
    {

        

        

        for i in 1..(Length(ps))
        {

            // Step 0
            ApplyPauliFromInt(PauliX, true, ps[i-1], p);
            
            // Step 1: 2XOR Gate with p and u[1] as controls and m as the target
            for num in 0..Length(p)-1
            {
                CCNOT(p[num],u[1],m[num]);
            }

            // Step 2: First XOR gate, then NOT gate with p as the control and m as the target
            for num in 0..Length(p)-1
            {
                CNOT(p[num],m[num]);
                X(m[num]);
            }

            // Step 3: nXOR Gate with m as the control and u[0] as the target
            Controlled X(m,u[0]);

            // Step 4: Controlled S^i Gate
            ControlledSi(p,u, i, Length(ps));

            // Step 5: nXOR Gate with m as the control and u[0] as the target
            Controlled X(m, u[0]);

            // Step 6: First NOT gate, then NOT gate with p as the control and m as the target
            for num in Length(p)-1..-1..0
            {
                X(m[num]);
                CNOT(p[num],m[num]);
            }

            // Step 7: 2XOR gate with p and u[1] as controls and m as the gate
            for num in Length(p)-1..-1..0
            {
                CCNOT(p[num],u[1],m[num]);
            }

            // Step 8: Clear the pattern register
            ApplyPauliFromInt(PauliX, true, ps[i-1], p);

        }
        
    }

    // Used in the CS^i step
    operation ControlledSi(p : Qubit[], u : Qubit[], i : Int, lenP: Int) : Unit is Adj+Ctl {
        let n = Length(p);

            let j = IntAsDouble(lenP + 1 - i);
            mutable angle = FindAngle(j);

            
            // Apply controlled Ry with u[0] as the control and u[1] as the target with the angle
            Controlled Ry([u[0]], (angle, u[1]));
    }

    // Sets angle to be between 0 and negative pi to ensure that the sin(angle/2) is negative and cos(angle/2) is negative
    function FindAngle(j: Double) : Double {
        //let angle = -2.0 * ArcSin(1.0 / Sqrt(j));
        let angle = -2.0 * ArcCos(Sqrt((j-1.)/j));

        Message($"{angle}");

        return angle;
    }



    operation RetrievalAlgorithm(i: Qubit[], m: Qubit[], c: Qubit) : Unit
    {
        // Step 0: H gate for c qubit
        H(c);

        // Step 1: First XOR gate, then NOT gate with m as the target and i as the control
        for num in 0..Length(i)-1
        {
            CNOT(i[num],m[num]);
            X(m[num]);
        }

        // Step 2: Unitary Exponential Operator
        ApplyUAndCU(m,c);

        // Step 3: First NOT gate, then XOR gate with m as the target and i as the control
        for num in Length(i)-1..-1..0
        {
            X(m[num]);
            CNOT(i[num],m[num]);
        }

        H(c);
    }

    operation U(q: Qubit, n: Int) : Unit is Adj+Ctl {
        X(q);
        R1(PI() / (2.0 * IntAsDouble(n)), q);
        X(q);
    } 

    operation CUSquared(q: Qubit, n: Int) : Unit is Adj+Ctl {
        Adjoint U(q,n);
        Adjoint U(q,n);
    } 

    operation ApplyUAndCU(m : Qubit[], c : Qubit) : Unit is Adj+Ctl {

        let n = Length(m);
        
        // Apply U First
        for j in 0..n-1 {
            
            U(m[j],n);

        }


        // CU^-2 Second
        for i in 0..n-1 {
            Controlled CUSquared([c], (m[i],n));
        }

        
    }

    // Derives x from f(x) retrieved using retrieval algorithm
    operation BVAlgorithm(i: Qubit[], m: Qubit[], c: Qubit, highDegree: Int) : Bool[]
    {
        mutable array = [];
        ApplyToEach(H,i);
        RetrievalAlgorithm(i,m,c);
        ApplyToEach(H,i);

        for ia in i
        {
            set array += [MResetZ(ia) == Zero ? true | false];
        }

        return array;
    }

    function FilterHighDegreeTerms(measured: Bool[], degreeThreshold: Int) : Bool[] {
        mutable filteredArray = [];

        let termDegree = CountOnes(measured);

        for idx in 0..Length(measured)-1 {

            if termDegree <= degreeThreshold {
                set filteredArray += [measured[idx]];
            }
        }

        return filteredArray;
    }

    function CountOnes(term: Bool[]) : Int {
        mutable count = 0;

        for val in term {
            if val {
                set count += 1;
            }
        }

        return count;
    }



    
    @EntryPoint()
operation QuantumMemoryAlgorithms() : Unit {
    Message("Initializing Quantum Memory...");

    // Test Case 1: Simple Input
    // Storage Registers
        use i = Qubit[3]; // Input Pattern Register
        use u = Qubit[2]; // Intermediate Register
        use m = Qubit[3]; // Memory Register (shared)
        

        // Retrieval Registers
        //use input = Qubit[2];
        //use c = Qubit(); // Control Register (for retrieval)

        

        
        X(u[1]);
        DumpMachine();

        let ps = [1,4];



        Message($"Storage Results");

        StorageAlgorithm(ps, i, u, m);
        DumpMachine();

        //X(input[0]);
        //X(i[1]); // Input is 1
        X(i[2]);

        Message($"Retrieval Results");
        
        RetrievalAlgorithm(i[1..Length(i)-1], m[1..Length(m)-1], u[0]);
        DumpMachine();

        // Reset all registers
        ApplyToEach(Reset, i);
        ApplyToEach(Reset, u);
        ApplyToEach(Reset, m);
        //Reset(c);


    

    Message("Quantum Memory Algorithm Tests Complete.");
}

}
