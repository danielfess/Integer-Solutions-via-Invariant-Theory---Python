from sympy import *
import os

f0 = Symbol('f0')
f1 = Symbol('f1')
f2 = Symbol('f2')
f3 = Symbol('f3')
f4 = Symbol('f4')
f5 = Symbol('f5')

s1 = Symbol('s1')
s2 = Symbol('s2')
s3 = Symbol('s3')
s4 = Symbol('s4')
s5 = Symbol('s5')

n11,n12,n13,n14,n15,n21,n22,n23,n24,n25,n31,n32,n33,n34,n35 = symbols('n11 n12 n13 n14 n15 n21 n22 n23 n24 n25 n31 n32 n33 n34 n35')

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

def matrix_add(A,B):
	"""Add matrices A and B of the same dimensions."""

	if len(A) != len(B) or len(A[0]) != len(B[0]):
		print('A and B are of different dimensions')
		return
	sum = []
	m = len(A)
	n = len(A[0])
	for i in range(m):
		sum.append([])
		for j in range(n):
			entry = A[i][j] + B[i][j]
			sum[i].append(entry)
	return sum

def matrix_scalar(A,k):
	"""Returns matrix k.A"""

	mult = []
	for i in range(len(A)):
		mult.append([])
		for j in range(len(A[0])):
			mult[i].append(k*A[i][j])
	return mult

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

Q = dict()
#Signed sub-pfaffs multiplied by 2 to make integral.

Q[5] = [[2*f0,f1,0,0],[f1,2*f2,f3,0],[0,f3,2*f4,f5],[0,0,f5,0]]
Q[4] = [[0,f0,0,0],[f0,2*f1,f2,0],[0,f2,2*f3,f4],[0,0,f4,2*f5]]
Q[1] = [[0,0,1,0],[0,-2,0,0],[1,0,0,0],[0,0,0,0]]
Q[2] = [[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]]
Q[3] = [[0,0,0,0],[0,0,0,1],[0,0,-2,0],[0,1,0,0]]


def det_lin_combo(s):
	mat = matrix_scalar(Q[1], s[0])
	for i in range(2,6):
		mat = matrix_add(mat,matrix_scalar(Q[i],s[i-1]))
	F = determinant(mat)
	return F

s = [s1,s2,s3,s4,s5]
F = det_lin_combo(s)
print(expand(F))

s0 = [s1,s2,s3,0,0]
F0 = det_lin_combo(s0)
print(expand(F0))

N = [[n11,n12,n13,n14,n15],[n21,n22,n23,n24,n25],[n31,n32,n33,n34,n35]]
sN = matrix_mult([[s1,s2,s3]],N)
G = expand(det_lin_combo(sN[0]))

indices = [(i1,i2,i3) for i1 in range(0,5) for i2 in range(0,5) for i3 in range(0,5) if i1 + i2 + i3 == 4]
coeffs = dict()
for i1,i2,i3 in indices:
	poly = G.collect(s1).coeff(s1,i1)
	poly = poly.collect(s2).coeff(s2,i2)
	coeffs[i1,i2,i3] = poly.collect(s3).coeff(s3,i3)

#file = open('subpfaff_det_GL5.txt','a')
#if os.stat('subpfaff_det_GL5.txt').st_size == 0:
#	file.write('#Determinant of linear combination of subpfaffians after GL5 action\n#(s1,s2,s3) index, coefficient\n')
#
#or index, coeff in coeffs.items():
#	file.write('{}, {}\n\n'.format(index,coeff))

