"""Program to print eligible values of kr(s), lr(s) for i,j,k."""

upper = [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5)]
upperno4 = [(1,2),(1,3),(1,5),(2,3),(2,5),(3,5)]
index = [1,2,3,4,5]
i = 4
j = 4
k = 1

d = [1,2,3,5]
count = 0

for l11 in d:
    e = d.copy()
    e.remove(l11)
    for l21 in e:
        f = e.copy()
        f.remove(l21)
        for k31 in f:
            g = f.copy()
            g.remove(k31)
            print(l11,l21,k31,g[0])

def copyremove(d,elts):
    """Makes a copy of list d, then removes each element appearing in the
    list elts from this copy.  Outputs the copy with elts removed.

    d: list
    elements: list

    output:list
    """
    
    copied = d.copy()
    for item in elts:
        copied.remove(item)
    return copied

def permutation(d):
    """"Outputs a list containing all permutations of the list d.
    
    d: list
    
    output: list of lists
    """
    
    permutations = []
    if len(d) > 1:
        for item in d:
            new = d.copy()
            new.remove(item)
            f = permutation(new)
            for perm in f:
                perm.append(item)
            permutations += f
    else:
        permutations = [d]
    return permutations

def valid_entries(d):
    """Given a list d of tuples, tests if d contains duplicates or
    reverse duplicates, if it contains any tuples of the form (k,k), if
    it contains (4,5) or (5,4), or if it contains either [(2,5),(3,4)] or
    [(1,4),(3,5)] - or the reverses of these.  In any of these cases,
    returns False. Otherwise, returns True.
    
    d: list of tuples
    
    output: bool
    """

    forbidden1 = {(2,5),(5,2),(3,4),(4,3)}
    forbidden2 = {(1,4),(4,1),(3,5),(5,3)}
    reverse = []
    newlist = d.copy()
    newlist.append((4,5))
    for item in newlist:
        newitem = item[::-1]
        reverse.append(newitem)
    set1 = set(newlist)
    set2 = set(reverse)
    set3 = set1.union(set2)
    if len(set3) < 2*len(newlist):
        return False
    elif len(set3.union(forbidden1)) == len(set3) or len(set3.union(forbidden2)) == len(set3):
        return False
    else:
        return True

print(valid_entries([(1,2),(2,1),(3,4)]))
print(valid_entries([(1,2),(2,5),(4,4)]))
print(valid_entries([(1,2),(2,5),(3,4),(4,5)]))
print(valid_entries([(1,2),(2,4),(3,4),(5,4)]))
print(valid_entries([(1,2),(2,4),(3,2),(1,5),(3,4)]))
print(valid_entries([(1,2),(2,5),(3,2),(1,5),(3,4)]))
print(valid_entries([(1,2),(5,2),(3,2),(1,5),(3,4)]))




count = 0
for (k31,l31) in upperno4:
    for (k32,l32) in upperno4:
        for (k3,l3) in upper:
            newupper = copyremove(upper,[(k3,l3)])
            for (k4,l4) in newupper:
                newupper1 = copyremove(upper,[(k31,l31)])
                for (k41,l41) in newupper1:
                    index1 = copyremove(index,[i,k31,l31])
                    for r in range(2):
                        l11 = index1[r]
                        l21 = index1[1-r]
                        index2 = copyremove(index,[j,k32,l32])
                        for s in range(2):
                            l12 = index2[s]
                            l22 = index2[1-s]
                            index3 = copyremove(index,[k3,l3])
                            for perm in permutation(index3):
                                k11 = perm[0]
                                l1 = perm[1]
                                l2 = perm[2]
                                index4 = copyremove(index,[k41,l41])
                                for perm1 in permutation(index4):
                                    k12 = perm1[0]
                                    k21 = perm1[1]
                                    l42 = perm1[2]
                                    index5 = copyremove(index,[k4,l4])
                                    for perm2 in permutation(index5):
                                        k2 = perm2[0]
                                        k22 = perm2[1]
                                        k42 = perm2[2]
                                        entry0 = [(k,l1),(k2,l2),(k3,l3),(k4,l4)]
                                        entry1 = [(k11,l11),(k21,l21),(k31,l31),(k41,l41)]
                                        entry2 = [(k12,l12),(k22,l22),(k32,l32),(k42,l42)]
                                        if valid_entries(entry0) and valid_entries(entry1) and valid_entries(entry2):
                                            print(entry0,entry1,entry2)
                                            count += 1
                                        
        print('done',k32,l32)

print('done',count,'terms')

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
        sign = (-1) ** (j % 2)
        total += sign * A[0][col] * subdet

    return total

determinant
            