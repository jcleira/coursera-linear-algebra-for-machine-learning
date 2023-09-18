import numpy as np

def MultiplyRow(M, row_num, row_num_multiple):
    M_new = M.copy()
    M_new[row_num] = M_new[row_num]*row_num_multiple
    return M_new

def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()
    M_new[row_num_2] = M_new[row_num_1]*row_num_1_multiple+M_new[row_num_2]
    return M_new

def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()
    M_new[[row_num_1,row_num_2]] = M_new[[row_num_2,row_num_1]]
    return M_new

A = np.array([
    [2, -1, 1, 1],
    [1, 2, -1, -1],
    [-1, 2, 2, 2],
    [1, -1, 2, 1]
    ], dtype=np.dtype(float))

b = np.array([6, 3, 14, 8], dtype=np.dtype(float))

print("System of equations:", A)
print("Results of equations:", b)

d = np.linalg.det(A)

print("Determinant of matrix A:", d)

x = np.linalg.solve(A, b)

print("Solution of system of equations:", x)

A_test = np.array([
        [1, -2, 3, -4],
        [-5, 6, -7, 8],
        [-4, 3, -2, 1],
        [8, -7, 6, -5]
    ], dtype=np.dtype(float))
print("Original matrix:")
print(A_test)

print("\nOriginal matrix after its third row is multiplied by -2:")
print(MultiplyRow(A_test,2,-2))

print("\nOriginal matrix after exchange of the third row with the sum of itself and first row multiplied by 4:")
print(AddRows(A_test,0,2,4))

print("\nOriginal matrix after exchange of its first and third rows:")
print(SwapRows(A_test,0,2))

b_col = b.reshape(4,1)
A_system = np.hstack((A, b_col))

print("\nStaged matrix of system of equations:\n", A_system)

A_system = SwapRows(A_system, 0, 1)
print("\nStaged matrix after exchange of its first and second rows:\n", A_system)

A_ref = A_system
A_ref = AddRows(A_ref, 0, 1, -2)
print("\nStaged matrix after second row is replaced by the sum of itself and first row multiplied by -2:\n", A_ref)
A_ref = AddRows(A_ref, 0, 2, 1)
print("\nStaged matrix after third row is replaced by the sum of itself and first row:\n", A_ref)
A_ref = AddRows(A_ref, 0, 3, -1)
print("\nStaged matrix after fourth row is replaced by the sum of itself and first row multiplied by -1:\n", A_ref)
A_ref = AddRows(A_ref, 2, 3, 1)
print("\nStaged matrix after fourth row is replaced by the sum of itself and third row:\n", A_ref)
A_ref = SwapRows(A_ref, 1, 3)
print("\nStaged matrix after exchange of its second and fourth rows:\n", A_ref)
A_ref = AddRows(A_ref, 2, 3, 1)
print("\nStaged matrix after fourth row is replaced by the sum of itself and third row:\n", A_ref)
A_ref = AddRows(A_ref, 1, 2, -4)
print("\nStaged matrix after third row is replaced by the sum of itself and second row multiplied by -4:\n", A_ref)
A_ref = AddRows(A_ref, 1, 3, 1)
print("\nStaged matrix after fourth row is replaced by the sum of itself and second row:\n", A_ref)
A_ref = AddRows(A_ref, 3, 2, 2)
print("\nStaged matrix after third row is replaced by the sum of itself and fourth row multiplied by 2:\n", A_ref)
A_ref = AddRows(A_ref, 2, 3, -8)
print("\nStaged matrix after fourth row is replaced by the sum of itself and third row multiplied by -8:\n", A_ref)
A_ref = MultiplyRow(A_ref, 3, -1/17)
print("\nStaged matrix after fourth row is multiplied by -1/17:\n", A_ref)

A_diag = A_ref
A_diag = AddRows(A_diag, 3, 2, -3)
print("\nStaged matrix after third row is replaced by the sum of itself and fourth row multiplied by -3:\n", A_diag)
A_diag = AddRows(A_diag, 3, 1, -3)
print("\nStaged matrix after second row is replaced by the sum of itself and fourth row multiplied by -3:\n", A_diag)
A_diag = AddRows(A_diag, 3, 0, 1)
print("\nStaged matrix after first row is replaced by the sum of itself and fourth row:\n", A_diag)
A_diag = AddRows(A_diag, 2, 1, -4)
print("\nStaged matrix after second row is replaced by the sum of itself and third row multiplied by -4:\n", A_diag)
A_diag = AddRows(A_diag, 2, 0, 1)
print("\nStaged matrix after first row is replaced by the sum of itself and third row:\n", A_diag)
A_diag = AddRows(A_diag, 1, 0, -2)
print("\nStaged matrix after first row is replaced by the sum of itself and second row multiplied by -2:\n", A_diag)


