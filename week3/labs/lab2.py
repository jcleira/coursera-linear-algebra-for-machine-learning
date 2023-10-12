import numpy as np

A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])
print("Matrix A (3 by 3):\n", A)

B = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 by 2):\n", B)

C = np.matmul(A, B)
print("Matrix C (3 by 2):\n", C)

D = A @ B
print("Matrix D (3 by 2):\n", D)

try:
    E = np.matmul(B, A)
except ValueError as err:
    print("Matrix multiplication not possible!", err)

try:
    F = B @ A
except ValueError as err:
    print("Matrix multiplication not possible!", err)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

np.matmul(x, y)
print("Dot product of x and y:", np.matmul(x, y))

np.dot(A, B)
print("Dot product of A and B:\n", np.dot(A, B))
