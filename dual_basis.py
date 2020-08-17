from sympy import *
import math

#Compute structure coefficients with respect to dual basis of canonical basis.

def determinant(A):
    """Given a square matrix A, compute its determinant.
    
    A: square matrix (in list form i.e. list of lists of numbers)
    
    output: same type as entries of A
    """
    
    total = 0

    if len(A) == 1:
        return A[0][0]

    for col in range(len(A)):
        Asub = A[1:]
        for j in range(len(A)-1):
            Asub[j] = Asub[j][:col] + Asub[j][col+1:]
        subdet = determinant(Asub)
        sign = (-1) ** (col % 2)
        total += sign * A[0][col] * subdet

    return total

def matrix_mult(A,B):
    """Computes the matrix product A B

    A: m x p matrix
    B: p x n matrix

    output: m x n matrix
    """

    m = len(A)
    p = len(B)
    n = len(B[0])
    AB = []
    for i in range(m):
        AB.append([])
        for j in range(n):
            total = 0
            for k in range(p):
                total += A[i][k] * B[k][j]
            AB[i].append(total)
    return AB


#Cubic ring calculations:

a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
d = Symbol('d')

M0 = [[1,0,0],[0,1,0],[0,0,1]]
M1 = [[0,-a*c,-a*d],[1,b,0],[0,-a,0]]
M2 = [[0,-a*d,-b*d],[0,0,d],[1,0,-c]]

B = [[3,b,-c],[b,b**2 - 2*a*c,-3*a*d],[-c,-3*a*d,c**2 - 2*b*d]]
A_integral = [[b**2*c**2 - 2*a*c**3 - 2*b**3*d + 4*a*b*c*d - 9*a**2*d**2, -b*c**2 + 2*b**2*d + 3*a*c*d, b**2*c - 2*a*c**2 - 3*a*b*d],
    [-b*c**2 + 2*b**2*d + 3*a*c*d, 2*c**2 - 6*b*d, 9*a*d - b*c],[b**2*c - 2*a*c**2 - 3*a*b*d, 9*a*d - b*c, 2*b**2 - 6*a*c]]
A = []
for i in range(3):
    A.append([])
    for j in range(3):
        A[i].append()
disc = b**2*c**2 - 4*a*c**3 - 4*b**3*d + 18*a*b*c*d - 27*a**2*d**2
disc_M0 = [[b**2*c**2 - 4*a*c**3 - 4*b**3*d + 18*a*b*c*d - 27*a**2*d**2,0,0],[0,b**2*c**2 - 4*a*c**3 - 4*b**3*d + 18*a*b*c*d - 27*a**2*d**2,0],[0,0,b**2*c**2 - 4*a*c**3 - 4*b**3*d + 18*a*b*c*d - 27*a**2*d**2]]

print(factor(matrix_mult(A,B)) == M0)
print(factor(matrix_mult(B,A)) == M0)

L0 = factor(matrix_mult(matrix_mult(B,M0),A))
L1 = factor(matrix_mult(matrix_mult(B,M1),A))
L2 = factor(matrix_mult(matrix_mult(B,M2),A))

print(L0)
print(L1)
print(L2)