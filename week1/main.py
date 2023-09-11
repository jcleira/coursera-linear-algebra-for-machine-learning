import numpy as np

A = np.array([
        [-1,3],
        [3,2]
    ], dtype=np.dtype(float))


b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

print(f"\nShape of A: {A.shape}")
print(f"Shape of b: {b.shape}")

x = np.linalg.solve(A, b)
print(f"\nSolution x: {x}")

d = np.linalg.det(A)
print(f"\nDeterminant of matrix A: {d:.2f}")

A_system = np.hstack((A, b.reshape(2,1)))

print (f"\nAugmented matrix: \n{A_system}")

A_system_res = A_system.copy()
A_system_res[1] = A_system_res[0] * 3 + A_system_res[1]
print (f"\nAugmented matrix: \n{A_system_res}")

A_system_res[1]= A_system_res[1] / 11
print (f"\nAugmented matrix: \n{A_system_res}")

AA = np.array([
    [-1, 3],
    [3, -9]
], dtype=np.dtype(float))

bb = np.array([7, -1], dtype=np.dtype(float))

det = np.linalg.det(AA)
print(f"\nDeterminant of matrix AA: {det:.2f}")

try:
    solution = np.linalg.solve(AA, bb)
    print(f"\nSolution x: {solution}")
except np.linalg.LinAlgError as err:
    print(f"\nError: {err}")

AAA = np.array([
    [-1, 3],
    [3, -9]
], dtype=np.dtype(float))

bbb = np.array([7, -21], dtype=np.dtype(float))

det = np.linalg.det(AAA)
print(f"\nDeterminant of matrix AAA: {det:.2f}")

try:
    solution = np.linalg.solve(AAA, bbb)
    print(f"\nSolution x: {solution}")
except np.linalg.LinAlgError as err:
    print(f"\nError: {err}")
