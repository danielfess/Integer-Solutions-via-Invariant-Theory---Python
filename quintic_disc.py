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

start = time.time()

theta = [[0,0,0,0,-f5/f0],[1,0,0,0,-f4/f0],[0,1,0,0,-f3/f0],[0,0,1,0,-f2/f0],[0,0,0,1,-f1/f0]]
identity = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
powers = [identity,theta,0,0,0]
for i in range(2,5):
	powers[i] = matrix_mult(theta,powers[i-1])

quintic_mult = dict()

for i in range(5):
	for j in range(5):
		for k in range(5):
			quintic_mult[(i,j,k)] = powers[i][k][j]

#Checking commutativity:
count = 0
for i in range(5):
	for j in range(5):
		for k in range(5):
			if quintic_mult[(i,j,k)] != quintic_mult[(j,i,k)]:
				count += 1
				print(i,j,k)
if count == 0:
	print('Multiplication is commutative')

basis_traces = [0]*5
for i in range(5):
	trace = 0
	for r in range(5):
		trace += quintic_mult[(i,r,r)]
	basis_traces[i] = trace

traces = []
for i in range(5):
	traces.append([])
	for j in range(5):
		trace = 0
		for k in range(5):
			trace += quintic_mult[(i,j,k)]*basis_traces[k]
		traces[i].append(trace)

middle = time.time()
print('Time to find traces:', (middle - start)/60)

disc = expand(f0**8 * determinant(traces))
print(disc)

with open('quintic_disc.txt','a') as f:
	if os.stat('quintic_disc.txt').st_size == 0:
		f.write('#Quintic ring discriminant\n')
	f.write('{}\n'.format(disc))

end = time.time()
print('Time to compute disc:', (end - middle)/60)