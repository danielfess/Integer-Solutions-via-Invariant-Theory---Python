#Program to compute the Dijk for four 5x5 skew-sym matrices
#coming from a binary quintic form.

from sympy import *
import time
import itertools
import os
from os import path

upper = [(1,2),(1,3),(1,4),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
upper_all = [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
index = [1,2,3,4,5]

def copyremove(d,elts):
    """Makes a copy of list d, then removes each item appearing in the
    sequence elts from this copy once.  Outputs the copy with elts removed.

    d: list
    elts: sequence (e.g. list, tuple,...)

    output:list
    """
    
    copied = d.copy()
    for item in set(elts):
        copied.remove(item)
    return copied

#Make copyremove a dict
indexremove_dict = {}
upperremove_dict = {}
specialupperremove_dict = {}

def indexremove(elts):

    if tuple(elts) in indexremove_dict:
        return indexremove_dict[tuple(elts)]
    indexremove_dict[tuple(elts)] = copyremove(index,elts)
    return indexremove_dict[tuple(elts)]

def upperremove(elts):

    if tuple(elts) in upperremove_dict:
        return upperremove_dict[tuple(elts)]
    upperremove_dict[tuple(elts)] = copyremove(upper,elts)
    return upperremove_dict[tuple(elts)]

def specialupperremove(elts):

    if tuple(elts) in specialupperremove_dict:
        return specialupperremove_dict[tuple(elts)]
    specialupperremove_dict[tuple(elts)] = copyremove(special_upper,elts)
    return specialupperremove_dict[tuple(elts)]

permutations_dict = dict()

def permutation(d):
    """"Outputs a list containing all permutations of the list d.
    
    d: list
    
    output: list of tuples
    """
    
    if tuple(d) in permutations_dict:
        return permutations_dict[tuple(d)]

    d_permutations = list()
    if len(d) > 1:
        for item in d:
            new = d.copy()
            new.remove(item)
            f = permutation(new)
            for perm in f:
                perm += item,
                d_permutations.append(perm)
    else:
        d_permutations = [tuple(d)]
    permutations_dict[tuple(d)] = d_permutations
    return d_permutations

permutation([1,2,3,4,5])

valid_dict = {}

def valid_entries(d):
    """Given a tuple d of tuples, tests if d contains duplicates or
    reverse duplicates, if it contains any tuples of the form (k,k), if
    it contains (1,5) or (5,1), or if it contains either [(1,4),(3,5)] or
    [(1,3),(2,5)] - or the reverses of these.  In any of these cases,
    returns False. Otherwise, returns True.
    
    d: sequence of tuples
    
    output: bool
    """

    if d in valid_dict:
        return valid_dict[d]

    forbidden1 = {(1,4),(4,1),(3,5),(5,3)}
    forbidden2 = {(2,5),(5,2),(3,1),(1,3)}
    reverse = []
    newtuple = d
    newtuple += (1,5),
    for item in newtuple:
        newitem = item[::-1]
        reverse.append(newitem)
    set1 = set(newtuple)
    set2 = set(reverse)
    set3 = set1.union(set2)
    if len(set3) < 2*len(newtuple):
        valid_dict[d] = False
        return False
    elif len(set3.union(forbidden1)) == len(set3) or len(set3.union(forbidden2)) == len(set3):
        valid_dict[d] = False
        return False
    else:
        valid_dict[d] = True
        return True

print(valid_entries(((1,2),(2,1),(3,4))))
print(valid_entries(((1,2),(2,5),(4,4))))
print(valid_entries(((1,2),(2,5),(3,4),(5,2))))
print(valid_entries(((1,2),(2,4),(3,4),(5,4))))
print(valid_entries(((1,2),(2,4),(3,2),(1,5),(3,4))))
print(valid_entries(((5,2),(3,2),(4,5),(1,3),(3,4))))
print(valid_entries(((1,2),(5,2),(3,2),(3,5),(3,4))))
print(valid_entries(((1, 1), (2, 2), (3, 5), (4, 5))))

upper_remove = [0,[(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)],[(1,3),(1,4),(3,4),(3,5),(4,5)],[(1,2),(1,4),(2,4),(2,5),(4,5)],[(1,2),(1,3),(2,3),(2,5),(3,5)],[(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]]
#Note: (1,5) removed from upper and upper_remove.

def indices(i,j,k):
    """Produces a listof the indices of all non-zero
    terms in the sum for c_{i,j}^k.  Each valid collection of indices is
    a 3-tuple of four 2-tuples.
    
    i,j,k: integers between 1 and 5
    
    output: list
    """

    start = time.time()
    count = 0
    indices = []
    for (k31,l31) in upper_remove[i]:
        for (k32,l32) in upper_remove[j]:
            for (k3,l3) in upper:
                newupper = upperremove([(k3,l3)])
                for (k4,l4) in newupper:
                    newupper1 = upperremove([(k31,l31)])
                    for (k41,l41) in newupper1:
                        index1 = indexremove((i,k31,l31))
                        for r in range(2):
                            l11 = index1[r]
                            l21 = index1[1-r]
                            index2 = indexremove((j,k32,l32))
                            for s in range(2):
                                l12 = index2[s]
                                l22 = index2[1-s]
                                index3 = indexremove((k3,l3))
                                for perm in permutation(index3):
                                    k11, l1, l2 = perm
                                    index4 = indexremove((k41,l41))
                                    for perm1 in permutation(index4):
                                        k12, k21, l42 = perm1
                                        index5 = indexremove((k4,l4))
                                        for perm2 in permutation(index5):
                                            k2, k22, k42 = perm2
                                            entry0 = ((k,l1),(k2,l2),(k3,l3),(k4,l4))
                                            entry1 = ((k11,l11),(k21,l21),(k31,l31),(k41,l41))
                                            entry2 = ((k12,l12),(k22,l22),(k32,l32),(k42,l42))
                                            if valid_entries(entry0) and valid_entries(entry1) and valid_entries(entry2):
                                                count += 1
                                                indices.append((entry0, entry1, entry2))
    print(entry0,entry1,entry2) 
    print('done:',count,'terms')
    end = time.time()
    print('Indices time:', (end - start)/60)
    return indices

def indices_iter(i,j,k):
    """Generator function defining an iterator which iterates through the indices of
    non-zero terms in the formula defining D_{i,j}^k.

    i,j,k: integers between 1 and 5

    yields: iterator of 3-tuples of four 2-tuples of int
    """

    start = time.time()
    perms1 = permutation(indexremove([i])).copy()
    removal = []
    for entry in perms1:
        if entry[2] > entry[3]:
            removal.append(entry)
    for entry in removal:
        perms1.remove(entry)
    removal = []

    perms2 = permutation(indexremove([j])).copy()
    for entry in perms2:
        if entry[2] > entry[3]:
            removal.append(entry)
    for entry in removal:
        perms2.remove(entry)
    removal = []

    perms3 = permutation(index).copy()
    for entry in perms3:
        if entry[3] > entry[4]:
            removal.append(entry)
    for entry in removal:
        perms3.remove(entry)

    indices_all = itertools.product(perms1,perms2,perms3,perms3,perms3)

    middle = time.time()
    print('middle',(middle-start)/60)

    while True:
        try:
            entry = next(indices_all)
            ((l11, l21, k31, l31), (l12, l22, k32, l32), (k11, l1, l2, k3, l3), (k12, k21, l42, k41, l41), (k2, k22, k42, k4, l4)) = entry
            entry0 = ((k,l1),(k2,l2),(k3,l3),(k4,l4))
            entry1 = ((k11,l11),(k21,l21),(k31,l31),(k41,l41))
            entry2 = ((k12,l12),(k22,l22),(k32,l32),(k42,l42))
            if valid_entries(entry0) and valid_entries(entry1) and valid_entries(entry2):
                yield (entry0, entry1, entry2)
        except StopIteration:
            break

def indices_iter_test(i,j,k):
    """Testing how long it takes to iterate through the iterator given by indices_iter(i,j,k)

    i,j,k: integers between 1 and 5

    output: None
    """

    start = time.time()
    count = 0
    for entry in indices_iter(i,j,k):
        count += 1
    print(count)
    end = time.time()
    print((end - start)/60)
    return


special_upper = [(1,2),(1,3),(1,4),(2,3),(2,5),(3,4),(3,5),(4,5)]
special_upper_remove = [0,[(2,3),(2,5),(3,4),(3,5),(4,5)],[(1,3),(1,4),(3,4),(3,5),(4,5)],[(1,2),(1,4),(2,5),(4,5)],[(1,2),(1,3),(2,3),(2,5),(3,5)],[(1,2),(1,3),(1,4),(2,3),(3,4)]]


def special_indices(i,j,k):
    """Produces a dictionary whose keys are the indices of all non-zero
    terms in the sum for c_{i,j}^k.  Each valid collection of indices is
    a 3-tuple of four 2-tuples.  Works for a binary form with only
    f0, f5 non-zero.
    
    i,j,k: integers between 1 and 5
    
    output: dict
    """

    start = time.time()
    count = 0
    indices = []
    for (k31,l31) in special_upper_remove[i]:
        for (k32,l32) in special_upper_remove[j]:
            for (k3,l3) in special_upper:
                newupper = specialupperremove([(k3,l3)])
                for (k4,l4) in newupper:
                    newupper1 = specialupperremove([(k31,l31)])
                    for (k41,l41) in newupper1:
                        index1 = indexremove((i,k31,l31))
                        for r in range(2):
                            l11 = index1[r]
                            l21 = index1[1-r]
                            index2 = indexremove((j,k32,l32))
                            for s in range(2):
                                l12 = index2[s]
                                l22 = index2[1-s]
                                index3 = indexremove((k3,l3))
                                for perm in permutation(index3):
                                    k11, l1, l2 = perm
                                    index4 = indexremove((k41,l41))
                                    for perm1 in permutation(index4):
                                        k12, k21, l42 = perm1
                                        index5 = indexremove((k4,l4))
                                        for perm2 in permutation(index5):
                                            k2, k22, k42 = perm2
                                            entry0 = ((k,l1),(k2,l2),(k3,l3),(k4,l4))
                                            entry1 = ((k11,l11),(k21,l21),(k31,l31),(k41,l41))
                                            entry2 = ((k12,l12),(k22,l22),(k32,l32),(k42,l42))
                                            if valid_entries(entry0) and valid_entries(entry1) and valid_entries(entry2):
                                                count += 1
                                                indices.append((entry0, entry1, entry2))
    print(entry0,entry1,entry2)
    end = time.time()
    print('done:', count, 'terms')
    print('Special indices time:', (end - start)/60)
    return indices

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

def reorder_sign(sequence):
    """Sorts a sequence, then returns the sorted sequence and the sign of
    the sorting permutation (of {1,...,n} where n = len(sequence)).
    
    sequence: list or tuple
    
    output: (reorder, sgn): (tuple, +1 or -1)
    """

    index = []
    for i in range(len(sequence)):
         index.append(i+1)
    track_sort = list(zip(sequence,index))
    track_sort.sort()
    reorder, perm = zip(*track_sort)
    sgn = sign(perm)
    return reorder, sgn

invt_dict = dict()

def invariant(A1,A2,A3,A4,invt_ind):
    """Computes the SL4 invariant (specified by invt_ind) of the
    quadruple of skew-sym matrices A1, A2, A3, A4.
    
    A1, A2, A3, A4: 5x5 skew-sym matrices = list of lists of size 5x5
    invt_ind: 4-tuple of 2-tuples of integers between 1 and 5

    output: same class as entries of matrices

    Note: The use of invt_dict here only gives the right answer when we fix
    A1,..,A4 throughout this script.
    """

    invt_ind_flip = tuple()
    flips = 0
    for tup in invt_ind:
        if tup in upper_all:
            invt_ind_flip += tup,
        else:
            invt_ind_flip += tup[::-1],
            flips += 1
    reorder, sgn = reorder_sign(invt_ind_flip)
    
    if reorder in invt_dict:
        return invt_dict[reorder]*sgn*((-1)**(flips % 2))
    matrix = [[A1[reorder[0][0]-1][reorder[0][1]-1],A1[reorder[1][0]-1][reorder[1][1]-1],A1[reorder[2][0]-1][reorder[2][1]-1],A1[reorder[3][0]-1][reorder[3][1]-1]],
              [A2[reorder[0][0]-1][reorder[0][1]-1],A2[reorder[1][0]-1][reorder[1][1]-1],A2[reorder[2][0]-1][reorder[2][1]-1],A2[reorder[3][0]-1][reorder[3][1]-1]],
              [A3[reorder[0][0]-1][reorder[0][1]-1],A3[reorder[1][0]-1][reorder[1][1]-1],A3[reorder[2][0]-1][reorder[2][1]-1],A3[reorder[3][0]-1][reorder[3][1]-1]],
              [A4[reorder[0][0]-1][reorder[0][1]-1],A4[reorder[1][0]-1][reorder[1][1]-1],A4[reorder[2][0]-1][reorder[2][1]-1],A4[reorder[3][0]-1][reorder[3][1]-1]]]
    det = determinant(matrix)
    invt_dict[reorder] = det
    return det*sgn*((-1)**(flips % 2))


print(determinant([[1,1,1],[0,1,3],[0,0,4]]))
#indices(4,4,1)

f0 = Symbol('f0')
f1 = Symbol('f1')
f2 = Symbol('f2')
f3 = Symbol('f3')
f4 = Symbol('f4')
f5 = Symbol('f5')


A1 = [[0,0,0,0,0],[0,0,-f0,-2*f1/5,0],[0,f0,0,-f2/10,0],[0,2*f1/5,f2/10,0,1],[0,0,0,-1,0]]
A2 = [[0,0,0,1,0],[0,0,-3*f1/5,-3*f2/5,0],[0,3*f1/5,0,-3*f3/10,-1],[-1,3*f2/5,3*f3/10,0,0],[0,0,1,0,0]]
A3 = [[0,0,-1,0,0],[0,0,-3*f2/10,-3*f3/5,1],[1,3*f2/10,0,-3*f4/5,0],[0,3*f3/5,3*f4/5,0,0],[0,-1,0,0,0]]
A4 = [[0,1,0,0,0],[-1,0,-f3/10,-2*f4/5,0],[0,f3/10,0,-f5,0],[0,2*f4/5,f5,0,0],[0,0,0,0,0]]

print(A1)
print(A2)
print(A3)
print(A4)

def sign(perm):
    """Returns the sign of a permutation of the set {1,..,n}.
    
    perm: list
    
    output: int: +1 or -1
    """

    if tuple(perm) in sign_dict:
        return sign_dict[tuple(perm)]
    n = len(perm)
    zeroes = []
    for i in range(n):
        zeroes.append(0)
    matrix = []
    for i in range(n):
        matrix.append(zeroes.copy())
        matrix[i][perm[i]-1] = 1
    det = determinant(matrix)
    sign_dict[tuple(perm)] = det
    return det

sign_dict = dict()

print(sign([1,3,5,4,2]))
print(sign([1,3,5,4,2]))

def term(A1,A2,A3,A4,index,i,j,k):
    """Computes the associated term in Dijk.
    
    A1, A2, A3, A4: 5x5 skew-sym matrices
    index: 3-tuple of four 2-tuples of integers between 1 and 5

    output: same type as entries of matrices
    """

    ((kdummy, l1),(k2,l2),(k3,l3),(k4,l4)),((k11,l11),(k21,l21),(k31,l31),(k41,l41)),((k12,l12),(k22,l22),(k32,l32),(k42,l42)) = index
    if kdummy != k:
        print("Need kdummy = k")
        return
    sign1 = sign([i,l11,l21,k31,l31])
    sign2 = sign([j,l12,l22,k32,l32])
    sign3 = sign([k11,l1,l2,k3,l3])
    sign4 = sign([k12,k21,k41,l41,l42])
    sign5 = sign([k2,k22,k4,k42,l4])
    signproduct = sign1 * sign2 * sign3 * sign4 * sign5
    invtproduct = invariant(A1,A2,A3,A4,index[0])*invariant(A1,A2,A3,A4,index[1])*invariant(A1,A2,A3,A4,index[2])
    return signproduct * invtproduct
    
    
test_index = ((1,2),(2,3),(3,4),(4,5)),((1,2),(2,3),(3,4),(4,5)),((1,2),(2,3),(3,4),(4,5))
print(term(A1,A2,A3,A4,test_index,1,1,1))

def Dijk(A1,A2,A3,A4,i,j,k):
    """Calculates D_{i,j}^k for the matrices A1,...,A4,
    using the list returned by the function indices.
    """

    sum_terms = indices(i,j,k)
    start = time.time()
    total = 0
    for sum_index in sum_terms:
        total += term(A1,A2,A3,A4,sum_index,i,j,k)
    end = time.time()
    print('Sum time:', (end - start)/60)
    return(total*32/2304)

def Dijk_iter(A1,A2,A3,A4,i,j,k):
    """Calculates D_{i,j}^k for the matrices A1,...,A4,
    using the generator function indices_iter.
    """

    start = time.time()
    sum_terms = indices_iter(i,j,k)
    total = 0
    for sum_index in sum_terms:
        total += term(A1,A2,A3,A4,sum_index,i,j,k)
    end = time.time()
    print('Total time:', (end - start)/60)
    return(total*32/2304)

def special_Dijk(A1,A2,A3,A4,i,j,k):
    """Calculates D_{i,j}^k for the matrices A1,...,A4 given
    by a binary form with f1 = f2 = f3 = f4 = 0,
    using the list returned by the function special_indices.
    """

    sum_terms = special_indices(i,j,k)
    start = time.time()
    total = 0
    for sum_index in sum_terms:
        total += term(A1,A2,A3,A4,sum_index,i,j,k)
    end = time.time()
    print('Sum time:', (end - start)/60)
    return(total*32/2304)

reorder, sgn = reorder_sign(((1,2),(2,4),(2,5),(2,3)))
print(reorder)
print(sgn)

start = time.time()

coefficients = dict()

filename = 'mult_coeffs.txt'
indices_done = []

if path.exists(filename) and os.stat(filename).st_size != 0:
	with open(filename, 'r') as f:
		line = f.readline()
		while line:
			if line.startswith('('):
				indices_done.append(line[:9])
			line = f.readline()

print(indices_done)
print(len(indices_done))

with open(filename, 'a') as f:
	if os.stat(filename).st_size == 0:
		f.write('#(i,j,k), D_{i,j}^k\n')
	for i in range(5):
		for j in range(5):
		    for k in range(5):
		        if str((i+1,j+1,k+1)) not in indices_done:
		            print(i+1,j+1,k+1)
		            coeff = Dijk(A1,A2,A3,A4,i+1,j+1,k+1)
		            coefficients[(i+1,j+1,k+1)] = factor(expand(coeff))
		            f.write('{} {}\n'.format((i+1,j+1,k+1),coefficients[(i+1,j+1,k+1)]))
		            f.flush()

for entry in coefficients.items():
	print(entry)

key = [(1,2,2),(2,1,1),(3,4,4),(4,3,3),(5,4,4)]
mult_coeffs = dict()

#for i in range(5):
#	for j in range(5):
#		for k in range(5):
#			if i == j:
#				if i == k:
#					mult_coeffs[(i+1,j+1,k+1)] = coefficients[(i+1,j+1,k+1)] - 2*coefficients[key[i]]
#				else:
#					mult_coeffs[(i+1,j+1,k+1)] = coefficients[(i+1,j+1,k+1)]
#			else:
#				if i == k:
#					mult_coeffs[(i+1,j+1,k+1)] = coefficients[(i+1,j+1,k+1)] - coefficients[key[j]]
#				elif j == k:
#					mult_coeffs[(i+1,j+1,k+1)] = coefficients[(i+1,j+1,k+1)] - coefficients[key[i]]
#				else:
#					mult_coeffs[(i+1,j+1,k+1)] = coefficients[(i+1,j+1,k+1)]

for entry in mult_coeffs.items():
	print(entry)


end = time.time()
print('Total time:', (end - start)/60)


#With (i,j,k) = (5,5,2), Dijk sums 271144 terms, result is quartic in fi
#with denominator 250 which cancels for integer-mat forms. Time: 2,18.

#With (i,j,k) = (5,5,5), Dijk sums 89542 terms, result is cubic in fi with
#denominator 75. Time: 2,2
#-4*(100*f0*f2*f5 - 20*f0*f3*f4 - 40*f1**2*f5 + 4*f1*f2*f4 + 2*f1*f3**2 - f2**2*f3)/75

#With (i,j,k) = (5,4,4), Dijk sums 70058 terms, result is cubic in fi with
#denominator 75.  Time: 2,2
#(325*f0*f2*f5 - 50*f0*f3*f4 - 160*f1**2*f5 + 25*f1*f2*f4 - 4*f1*f3**2 - f2**2*f3)/75

#d544 = 0, so d555 = D555 - 2*D544
#d555 = -2*(175*f0*f2*f5 - 30*f0*f3*f4 - 80*f1**2*f5 + 11*f1*f2*f4 - f2**2*f3)/25

#With (i,j,k) = (1,1,2), Dijk sums 269668 terms, result is quartic in fi
#with denominator 250 which cancels for integer-mat forms.

#(i,j,k) = (2,2,5), 1872 terms, result is -10*f5. Time: 1,0
#(i,j,k) = (2,2,1), 1440 terms, result is -2*f4. Time: 1,0
#(i,j,k) = (2,2,3), 9152 terms, result is (5*f2*f5 - f3*f4)/5. Time: 1,0
#(i,j,k) = (2,2,4), 10448 terms, result is 2*(5*f3*f5 - 2*f4**2)/5. Time: 1,0
#(i,j,k) = (2,2,2), 12832 terms, result is 2*(10*f1*f5 - f2*f4)/15. Time: 1,0
#(i,j,k) = (2,4,2), 12304 terms, result is -2*(10*f0*f4 - 7*f1*f3 + 3*f2**2)/15. Time: 1,0
#(i,j,k) = (2,1,1), 15438 terms, result is -(20*f1*f5 - 8*f2*f4 + 3*f3**2)/15. Time: 1,0
#d211 = 0, so d222 = D222 - 2*D211 = 2*(10*f1*f5 - 3*f2*f4 + f3**2)/5


