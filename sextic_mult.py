#Read Dijk from mult_coeffs.txt and use to check if S_f is a ring of the correct discriminant.

from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import os
from os import path
import time

f0 = Symbol('f0')
f1 = Symbol('f1')
f2 = Symbol('f2')
f3 = Symbol('f3')
f4 = Symbol('f4')
f5 = Symbol('f5')

g0 = Symbol('g0')
g1 = Symbol('g1')
g2 = Symbol('g2')
g3 = Symbol('g3')
g4 = Symbol('g4')
g5 = Symbol('g5')

filename = 'mult_coeffs.txt'
file = open(filename)
invariants = dict()
for line in file:
	if line.startswith('('):
		i,j,k = int(line[1]), int(line[4]), int(line[7])
		formula = parse_expr(line[10:])
		invariants[(i,j,k)] = formula

count = 0
total = 0
for i,j,k in invariants:
	total += 1
	if invariants[(i,j,k)] != invariants[(j,i,k)]:
		count += 1
print(total,'invariants')
if count == 0:
	print('The invariants Dijk are symmetric in i,j')

#Normalise beta_2, beta_3, beta_4 so that beta_3^2 = 4 beta_2 beta_4,
#and make beta_1, beta_5 trace-free:
key = [0,(1,2,2),(2,4,4),(3,3,3),(4,2,2),(5,4,4)]
mult_coeffs = dict()
mult_coeffs[key[2]] = invariants[(3,3,4)]/4
mult_coeffs[key[4]] = invariants[(3,3,2)]/4
mult_coeffs[key[3]] = 4*invariants[(2,4,3)]
mult_coeffs[key[1]] = -(-26*f0*f3*f5 + 64*f0*f4**2/5 + 4*f1*f2*f5 - 2*f1*f3*f4 + 8*f2**2*f4/25 + 2*f2*f3**2/25)/6
mult_coeffs[key[5]] = -(-26*f0*f2*f5 + 4*f0*f3*f4 + 64*f1**2*f5/5 - 2*f1*f2*f4 + 8*f1*f3**2/25 + 2*f2**2*f3/25)/6

for i,j,k in invariants:
	if i == j:
		if i == k:
			if i != 3:
				mult_coeffs[(i,i,i)] = invariants[(i,i,i)] - 2*invariants[key[i]] + 2*mult_coeffs[key[i]]
			if i == 3:
				continue
		else:
			mult_coeffs[(i,i,k)] = invariants[(i,i,k)]
	else:
		if i == k:
			if j != 3:
				mult_coeffs[(i,j,i)] = invariants[(i,j,i)] - invariants[key[j]] + mult_coeffs[key[j]]
			else:
				mult_coeffs[(i,3,i)] = mult_coeffs[(3,3,3)]/2 - invariants[(3,3,3)]/2 + invariants[(i,3,i)]
		elif j == k:
			if i != 3:
				mult_coeffs[(i,j,j)] = invariants[(i,j,j)] - invariants[key[i]] + mult_coeffs[key[i]]
			else:
				mult_coeffs[(3,j,j)] = mult_coeffs[(3,3,3)]/2 - invariants[(3,3,3)]/2 + invariants[(3,j,j)]
		else:
			mult_coeffs[(i,j,k)] = invariants[(i,j,k)]

not_i = [0,2,3,4,5,1]

for i in range(1,6):
	for j in range(1,6):
		k = not_i[i]
		constant_term = 0
		for r in range(1,6):
			constant_term += mult_coeffs[(j,k,r)]*mult_coeffs[(r,i,k)] - mult_coeffs[(i,j,r)]*mult_coeffs[(r,k,k)]
		mult_coeffs[(i,j,0)] = factor(expand(constant_term))

for i in range(6):
	for k in range(6):
		if i == k:
			mult_coeffs[(i,0,k)] = 1
			mult_coeffs[(0,i,k)] = 1
		else:
			mult_coeffs[(i,0,k)] = 0
			mult_coeffs[(0,i,k)] = 0

#Checking integrality/content for integer-matrix f:
print('integrality/content calculations')
mult_coeffs_intmat = dict()
for i,j,k in mult_coeffs:
	expr = factor(expand(mult_coeffs[(i,j,k)]))
	if i != 0 and j!= 0:
		expr = factor(expand(expr.subs([(f0,g0),(f1,5*g1),(f2,10*g2),(f3,10*g3),(f4,5*g4),(f5,g5)])))
	if i!=0 and i <= j:
		print((i,j,k),expr)
	mult_coeffs_intmat[(i,j,k)] = expr
print('integrality/content calcs done')

#Check commutativity:
count = 0
count_intmat = 0
total = 0
for i in range(6):
	for j in range(6):
		for k in range(6):
			total += 1
			if mult_coeffs[(i,j,k)] != mult_coeffs[(j,i,k)]:
				print(i,j,k)
				count += 1
			if mult_coeffs_intmat[(i,j,k)] != mult_coeffs_intmat[(j,i,k)]:
				print(i,j,k)
				count_intmat += 1
print(total, 'mult coeffs')
if count == 0:
	print('Multiplication is commutative')
if count_intmat == 0:
	print('No problems for int mat either')

matrices = [0]*6
for i in range(6):
	matrices[i] = []
	for k in range(6):
		matrices[i].append([])
		for j in range(6):
			matrices[i][k].append(mult_coeffs[(i,j,k)])

matrices_intmat = [0]*6
for i in range(6):
	matrices_intmat[i] = []
	for k in range(6):
		matrices_intmat[i].append([])
		for j in range(6):
			matrices_intmat[i][k].append(mult_coeffs_intmat[(i,j,k)])


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

#Check associativity:
count = 0
count_intmat = 0
total = 0
for i in range(6):
	for j in range(6):
			for k in range(6):
				i_jk = []
				ij_k = []
				i_jk_intmat = []
				ij_k_intmat = []
				for s in range(6):
					i_jk.append(0)
					ij_k.append(0)
					i_jk_intmat.append(0)
					ij_k_intmat.append(0)
					for r in range(6):
						i_jk[s] += mult_coeffs[(j,k,r)]*mult_coeffs[(i,r,s)]
						ij_k[s] += mult_coeffs[(i,j,r)]*mult_coeffs[(r,k,s)]
						i_jk_intmat[s] += mult_coeffs_intmat[(j,k,r)]*mult_coeffs_intmat[(i,r,s)]
						ij_k_intmat[s] += mult_coeffs_intmat[(i,j,r)]*mult_coeffs_intmat[(r,k,s)]
					if expand(i_jk[s]) != expand(ij_k[s]):
						count += 1
						print((i,j,k,s))
					if expand(i_jk_intmat[s]) != expand(ij_k_intmat[s]):
						count_intmat += 1
						print((i,j,k,s))
				total += 1
print(total,'associativity checks')
if count == 0:
	print('Multiplication is associative')
if count_intmat == 0:
	print('No problems for int mat either')

#Check discriminant:
basis_traces = [0]*6
for i in range(6):
	for j in range(6):
		basis_traces[i] += matrices[i][j][j]

for i in range(6):
	print(i,basis_traces[i])

traces_matrix = []
for i in range(6):
	traces_matrix.append([])
	for j in range(6):
		traces_matrix[i].append(0)
		for k in range(6):
			traces_matrix[i][j] += mult_coeffs[(i,j,k)]*basis_traces[k]

#Calculating how renormalizing by a covariant changes the V_IJK:
t = {}
t[2] = basis_traces[2]
t[3] = basis_traces[3]
t[4] = basis_traces[4]
d = [(1,1),(1,5),(4,5),(3,5),(1,4)]
renorm = {}
for (i,j) in d:
	renorm[(i,j)] = expand(mult_coeffs[(i,j,2)]*t[2] + mult_coeffs[(i,j,3)]*t[3] + mult_coeffs[(i,j,4)]*t[4])
	print((i,j),renorm[(i,j)])
	print((i,j),mult_coeffs[(i,j,0)])
d2 = [(4,4),(3,4),(3,3),(2,4)]
for (i,j) in d2:
	renorm[(i,j)] = expand(mult_coeffs[(i,j,2)]*t[2] + mult_coeffs[(i,j,3)]*t[3] + mult_coeffs[(i,j,4)]*t[4])
	print((i,j),renorm[(i,j)])
	print((i,j),expand(t[i]*t[j]))
	print((i,j),mult_coeffs[(i,j,0)])

print('I = {1,5}, J = {2,3,4}, K = 0, c1 = -2 c2')
print(expand(-2*38*mult_coeffs[(1,4,0)] + 38*mult_coeffs[(3,5,0)] + 5*(-2)*renorm[(1,4)] + 5*renorm[(3,5)]))

print('I = {1,5}, J = {2,3,4}, K = 0, c1 = c2')
print(expand(mult_coeffs[(1,4,0)] + mult_coeffs[(3,5,0)] - renorm[(1,4)]/2 - renorm[(3,5)]/2))

print('I = J = {2,3,4}, K = 0')
l = -3/10
a = 1
b = 2
print(expand(l**2*(a*t[3]**2 + b*t[2]*t[4]) + l*(a*renorm[(3,3)] + b*renorm[(2,4)]) - (a*mult_coeffs[(3,3,0)] + b*mult_coeffs[(2,4,0)])))

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

#Calculating and storing discriminant:

#start = time.time()
#discriminant = expand(determinant(traces_matrix))
#with open('sextic_disc.txt','a') as f:
#	if os.stat('sextic_disc.txt').st_size == 0:
#		f.write('#Discriminant of ring\n')
#	f.write('{}\n'.format(discriminant))
#end = time.time()
#print('Time:',(end - start)/60)

#Computing some norm forms:
print('Norm forms:')

x = Symbol('x')
y = Symbol('y')

xb1 = matrix_scalar(matrices_intmat[1],x)
yb5 = matrix_scalar(matrices_intmat[5],y)

x2b2 = matrix_scalar(matrices_intmat[2],x**2)
xyb3 = matrix_scalar(matrices_intmat[3],x*y)
y2b4 = matrix_scalar(matrices_intmat[4],y**2)

n1 = matrix_add(xb1,yb5)
n2 = matrix_add(x2b2,matrix_add(xyb3,y2b4))

norm1 = factor(expand(determinant(matrices_intmat[1])))
norm2 = factor(expand(determinant(matrices_intmat[2])))

print(norm1)
print(norm2)