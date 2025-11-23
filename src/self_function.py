from ma1522 import Matrix

def basis_of(T, V=None, verbosity=0):
    """Given a set T of vectors in vector space V, return True if T is a basis for V.
    
    Args:
        T: A Matrix where each column is a vector in the set T
        V: A Matrix where each column spans the vector space V.
           If None, defaults to R^n where n is the number of rows in T.
        verbosity: Level of output detail (0=none, 1=summary, 2=detailed)
        
    Returns:
        bool: True if T is a basis for V, False otherwise
    """
    if not isinstance(T, Matrix):
        raise TypeError("T must be a Matrix object")
    
    # If V is not provided, default to R^n (standard basis)
    if V is None:
        n = T.rows
        # Create identity matrix as standard basis for R^n
        V = Matrix.eye(n)
        if verbosity >= 1:
            print(f"✓ Vector space V not provided, defaulting to R^{n}")
    
    if not isinstance(V, Matrix):
        raise TypeError("V must be a Matrix object")
    
    # Check dimensions match
    if T.rows != V.rows:
        raise ValueError("T and V must have the same number of rows")
    
    if verbosity >= 1:
        print("="*60)
        print("Checking if T is a basis for V")
        print("="*60)
    
    # Check if T has the right number of vectors (must equal dimension of V)
    # The dimension of V is the rank of V
    dim_V = V.rank()
    num_vectors_T = T.cols
    
    if verbosity >= 1:
        print(f"\n1. Dimension Check:")
        print(f"   - dim(V) = rank(V) = {dim_V}")
        print(f"   - Number of vectors in T = {num_vectors_T}")
    
    # A basis must have exactly dim(V) vectors
    if num_vectors_T != dim_V:
        if verbosity >= 1:
            print(f"   ✗ FAIL: T has {num_vectors_T} vectors but needs exactly {dim_V} vectors")
            print(f"\n{'='*60}")
            print(f"Result: T is NOT a basis for V")
            print(f"Reason: Wrong number of vectors (need {dim_V}, have {num_vectors_T})")
            print(f"{'='*60}")
        return False
    
    if verbosity >= 1:
        print(f"   ✓ PASS: T has the correct number of vectors ({dim_V})")
    
    # Check if vectors in T are linearly independent
    if verbosity >= 1:
        print(f"\n2. Linear Independence Check:")
    
    is_independent = T.is_linearly_independent(colspace=True, verbosity=0)
    
    if not is_independent:
        if verbosity >= 1:
            print(f"   ✗ FAIL: Vectors in T are linearly dependent")
            if verbosity >= 2:
                print(f"\n   Computing RREF to show dependence:")
                rref = T.rref()
                print(f"   RREF(T) =")
                print(f"   {rref}")
            print(f"\n{'='*60}")
            print(f"Result: T is NOT a basis for V")
            print(f"Reason: Vectors are linearly dependent")
            print(f"{'='*60}")
        return False
    
    if verbosity >= 1:
        print(f"   ✓ PASS: Vectors in T are linearly independent")
        if verbosity >= 2:
            rref = T.rref()
            print(f"\n   RREF(T) shows pivot in each column:")
            print(f"   {rref}")
    
    # Check if T spans V (i.e., every vector in V can be written as linear combination of T)
    # This is equivalent to checking if rank([T | V]) == rank(T) == dim(V)
    if verbosity >= 1:
        print(f"\n3. Spanning Check:")
        print(f"   - Checking if T spans V")
    
    combined = T.row_join(V)
    rank_combined = combined.rank()
    
    if verbosity >= 2:
        print(f"   - rank(T) = {T.rank()}")
        print(f"   - rank([T | V]) = {rank_combined}")
        print(f"   - If rank([T | V]) = rank(T), then T spans V")
    
    if rank_combined != dim_V:
        if verbosity >= 1:
            print(f"   ✗ FAIL: T does not span V")
            print(f"   rank([T | V]) = {rank_combined} ≠ {dim_V} = dim(V)")
            print(f"\n{'='*60}")
            print(f"Result: T is NOT a basis for V")
            print(f"Reason: T does not span V")
            print(f"{'='*60}")
        return False
    
    if verbosity >= 1:
        print(f"   ✓ PASS: T spans V (rank([T | V]) = rank(T) = {dim_V})")
        print(f"\n{'='*60}")
        print(f"✓ Result: T IS a basis for V")
        print(f"{'='*60}")
        print(f"\nProof Summary:")
        print(f"  1. T has exactly {dim_V} vectors (dimension of V)")
        print(f"  2. The vectors in T are linearly independent")
        print(f"  3. T spans V")
        print(f"Therefore, T forms a basis for V. ∎")
        print(f"{'='*60}")
    
    return True