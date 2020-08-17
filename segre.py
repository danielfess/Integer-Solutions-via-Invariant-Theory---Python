from sympy import *
import math

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

print(determinant([[1]]))
print(determinant([[1,0],[0,1]]))
print(determinant([[1,1],[3,2]]))

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

A = [[1,2],[1,3]]
B = [[0,-1,2],[1,1,0]]
print(matrix_mult(A,B))

y1 = Symbol('y1')
y2 = Symbol('y2')
y3 = Symbol('y3')
y4 = Symbol('y4')
y5 = Symbol('y5')

z1 = Symbol('z1')
z2 = Symbol('z2')
z3 = Symbol('z3')
z4 = Symbol('z4')
z5 = Symbol('z5')

f0 = Symbol('f0')
f1 = Symbol('f1')
f2 = Symbol('f2')
f3 = Symbol('f3')
f4 = Symbol('f4')
f5 = Symbol('f5')

#A is from \Phi:
A = [0]*4

A[0] = [[0,0,0,0,0],[0,0,-f0,0,0],[0,f0,0,0,0],[0,0,0,0,1],[0,0,0,-1,0]]
A[1] = [[0,0,0,1,0],[0,0,-f1,-f2,0],[0,f1,0,0,-1],[-1,f2,0,0,0],[0,0,1,0,0]]
A[2] = [[0,0,-1,0,0],[0,0,0,-f3,1],[1,0,0,-f4,0],[0,f3,f4,0,0],[0,-1,0,0,0]]
A[3] = [[0,1,0,0,0],[-1,0,0,0,0],[0,0,0,-f5,0],[0,0,f5,0,0],[0,0,0,0,0]]

#A_prime is from \Phi', i.e. integer-matrix version:
A_prime = [0]*4

A_prime[0] = [[0,0,0,0,0],[0,0,-f0,-2*f1/5,0],[0,f0,0,-f2/10,0],[0,2*f1/5,f2/10,0,1],[0,0,0,-1,0]]
A_prime[1] = [[0,0,0,1,0],[0,0,-3*f1/5,-3*f2/5,0],[0,3*f1/5,0,-3*f3/10,-1],[-1,3*f2/5,3*f3/10,0,0],[0,0,1,0,0]]
A_prime[2] = [[0,0,-1,0,0],[0,0,-3*f2/10,-3*f3/5,1],[1,3*f2/10,0,-3*f4/5,0],[0,3*f3/5,3*f4/5,0,0],[0,-1,0,0,0]]
A_prime[3] = [[0,1,0,0,0],[-1,0,-f3/10,-2*f4/5,0],[0,f3/10,0,-f5,0],[0,2*f4/5,f5,0,0],[0,0,0,0,0]]

y = [[y1],[y2],[y3],[y4],[y5]]
z = [z1,z2,z3,z4,z5]

Ay = [0]*4
for i in range(4):
    Ay[i] = matrix_mult(A[i],y)

segre_matrix = []
for i in range(4):
    segre_matrix.append([])
    for j in range(5):
        segre_matrix[i].append(Ay[i][j][0])
segre_matrix.append(z)

print(segre_matrix)
F = factor(determinant(segre_matrix))
print("Bilinear factor x Segre cubic:")
print(F)



mat = [[0,y4,-y3,y2,z1],[-y3,0,y5,-y1,z2],[y2,-y5,y1,y4,z3],[y5,-y1,0,-y3,z4],[-y4,y3,-y2,0,z5]]
poly = determinant(mat)
print(factor(poly))

def segre_cubic(a,b,c,d,e):

    return (a**2*d - a*b**2 - a*c*e + b*c*d + b*e**2 - c**3 + d**2*e)

def segre_partials(a,b,c,d,e):

    return (2*a*d - b**2 - c*e, -2*a*b + c*d + e**2, -a*e + b*d - 3*c**2, a**2 + b*c + 2*d*e, -a*c + 2*b*e + d**2)

t1 = Symbol('t1')
t2 = Symbol('t2')

f0 = 1
f5 = -1

num = segre_cubic(-2*f0*(t1**2+t2**2),2*f0**2*(t1**4+t2**4),3*f0*f5,-2*f0*f5*(t1+t2),2*f0*(t1**3+t2**3))
print(factor(num))

partials = segre_partials(-2*f0*(t1**2+t2**2),2*f0**2*(t1**4+t2**4),3*f0*f5,-2*f0*f5*(t1+t2),2*f0*(t1**3+t2**3))
print(factor(partials))

print('Using basis of ring to get trace-free basis - as per Seok Hyeong email')
num = segre_cubic(f0*f5*(t1**3+t2**3),-f5*(t1+t2),f5,f0*(t1**4+t2**4),-f0*f5*(t1**2+t2**2))
print(factor(num))

partials = segre_partials(f0*f5*(t1**3+t2**3),-f5*(t1+t2),f5,f0*(t1**4+t2**4),-f0*f5*(t1**2+t2**2))
print(factor(partials))

a1 = (-1 + math.sqrt(5))/2
a2 = (1 + math.sqrt(5))/2

b1 = (-1 - math.sqrt(5))/2
b2 = (1 - math.sqrt(5))/2

num1 = segre_cubic(a1,a2,1,a2,-a1)
num2 = segre_cubic(b1,b2,1,b2,-b1)
print(num1)
print(num2)

num3 = segre_partials(a1,a2,1,a2,-a1)
num4 = segre_partials(b1,b2,1,b2,-b1)
print(num3)
print(num4)
