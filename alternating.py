#Checking that construction of alternating form is correct

from sympy import *

b11 = Symbol('b11')
b12 = Symbol('b12')
b13 = Symbol('b13')
b14 = Symbol('b14')
b15 = Symbol('b15')
b16 = Symbol('b16')
b21 = Symbol('b21')
b22 = Symbol('b22')
b23 = Symbol('b23')
b24 = Symbol('b24')
b25 = Symbol('b25')
b26 = Symbol('b26')
b31 = Symbol('b31')
b32 = Symbol('b32')
b33 = Symbol('b33')
b34 = Symbol('b34')
b35 = Symbol('b35')
b36 = Symbol('b36')
b41 = Symbol('b41')
b42 = Symbol('b42')
b43 = Symbol('b43')
b44 = Symbol('b44')
b45 = Symbol('b45')
b46 = Symbol('b46')
b51 = Symbol('b51')
b52 = Symbol('b52')
b53 = Symbol('b53')
b54 = Symbol('b54')
b55 = Symbol('b55')
b56 = Symbol('b56')


def transpose(A):
	A_t = []
	m = len(A)
	n = len(A[0])
	for i in range(n):
		A_t.append([])
		for j in range(m):
			A_t[i].append(A[j][i])
	return A_t

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

def submatrix(A,k,l):

	n = len(A)
	sub = []
	row = 0
	for i in range(n):
		if i != k:
			sub.append([])
			for j in range(n):
				if j != l:
					sub[row].append(A[i][j])
			row += 1
	return sub


def cofactor(A):

	cof = []
	n = len(A)
	for i in range(n):
		cof.append([])
		for j in range(n):
			A_sub = submatrix(A,j,i)
			det = expand(determinant(A_sub))
			cof[i].append((-1)**(i+j)*det)
	return cof

beta = [[1,1,1,1,1,1],[b11,b12,b13,b14,b15,b16],[b21,b22,b23,b24,b25,b26],[b31,b32,b33,b34,b35,b36],[b41,b42,b43,b44,b45,b46],[b51,b52,b53,b54,b55,b56]]
beta_t = transpose(beta)
traces = matrix_mult(beta,beta_t)
print(traces)
cof = cofactor(traces)
beta_dual_scaled = matrix_mult(cof,beta)

def g(u,v):

	A = [[1,1,1],[u[0]+u[1], u[2]+u[5], u[3]+u[4]], [v[0]+v[1], v[2]+v[5], v[3]+v[4]]]
	return determinant(A)

def f(x,y,z):

	A = [[x[0]-x[1], x[2]-x[5], x[3]-x[4]], [y[0]-y[1], y[2]-y[5], y[3]-y[4]], [z[0]-z[1], z[2]-z[5], z[3]-z[4]]]
	return 3*determinant(beta)**3*determinant(A)


print(beta)
print(beta_t)
print(traces)
print(cof)
#print(factor(beta_dual_scaled[1]))
#print(factor(beta_dual_scaled[2]))
print(beta[3])
print(beta[4])
print(beta[5])

LHS = g(beta_dual_scaled[1],beta_dual_scaled[2])
RHS = f(beta[3],beta[4],beta[5])
print(expand(LHS) == expand(RHS))