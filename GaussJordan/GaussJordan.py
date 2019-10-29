import numpy as np

ITERATION_LIMIT = 1000

# initialize the matrix
A = np.array([[4., 1., -1., 1.],
              [1., 4., -1., -1.],
              [-1., -1., 5., 1.],
              [1., -1., 1., 3.]])
# initialize the RHS vector
b = np.array([-2., -1., 0., 1.])

# prints the system
print("System:")
for i in range(A.shape[0]):
    row = ["{}*x{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
    print(" + ".join(row), "=", b[i])
print()

x = np.zeros_like(b)
for it_count in range(ITERATION_LIMIT):
    print("Current solution:", x)
    x_new = np.zeros_like(x)

    for i in range(A.shape[0]):
        s1 = np.dot(A[i, :i], x[:i])
        s2 = np.dot(A[i, i + 1:], x[i + 1:])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]

    if np.allclose(x, x_new, atol=0.000001, rtol=0.):
        break

    x = x_new

print("Solution:")
print(x)
error = np.dot(A, x) - b
print("Error:")
print(error)
