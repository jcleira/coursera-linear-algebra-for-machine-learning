import numpy as np

A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])
print("Matrix A (3 by 3):\n", A)

B = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 by 2):\n", B)

C = np.matmul(A, B)

print("Matrix C (3 by 2):\n", C)

D = A @ B

print("Matrix D (3 by 2):\n", D)
