#Count preimages for the solution X=8, Y=8, Z=8, k=7 to Y^2 = X^3 + k Z^2.

def count_preimages(X,Y,Z,k):
    """Counts preimages under integer-matrix covariant map for an integer
    solution (X,Y,Z,k) to Y^2 = X^3 + kZ^2.
    
    X,Y,Z,k: integers, with Z non-zero
    """
    
    print('\n')
    print('Count preimages for:')
    print('X','Y','Z','k')
    print(X,Y,Z,k)
    
    if Y**2 != X**3 + k*Z**2:
        print('Need Y**2 = X**3 + k*Z**2')
        return

    D = max_divisor(X,Y,Z)
    print('D = ',D)
    X1 = X//D**2
    Y1 = Y//D**3
    Z1 = Z//D**3
    preimages = 0
    
    print('lambda2_ | preimages')
    for i in range(D):
        if D%(i+1) == 0:
            lambda2_ = i + 1
            n = solution(lambda2_,X1,Y1,Z1)
            print(lambda2_,'|',n)
            preimages += n
    print('Total',preimages)

def solution(lambda2_,X1,Y1,Z1):
    """Returns the number of solutions modulo Z1*lambda2_**2 of the pair
    of equations (B**2 - X1) % Z1*lambda2_ == 0 and
    (-B**3 + 3*B*X1 - 2*Y1) % Z1**2*lambda2_**3 == 0.
    """
    count = 0
    for i in range(Z1*lambda2_**2):
        if ((i**2 - X1) % (Z1*lambda2_) == 0 and
            (-i**3 + 3*i*X1 - 2*Y1) % (Z1**2*lambda2_**3) == 0):
            count += 1
#            print('i = ',i)
    return count

print(solution(2,2,1,1))

def max_divisor(X,Y,Z):
    """Returns max pos int D such that D**3 divides Y and Z,
    and D**2 divides X.

    X,Y,Z: integers, with Z non-zero
    """
    
    i = 1
    while i**3 <= abs(Z):
        if (Y % i**3 == 0 and Z % i**3 == 0 and X % i**2 == 0):
            D = i
        i += 1
    return D

print(max_divisor(8,8,8))
print(max_divisor(36,216,1080))
print(max_divisor(16,64,16))
print(max_divisor(16,64,64))

count_preimages(8,8,8,-7)
count_preimages(8,8,8,1)
count_preimages(4,4,1,-48)
count_preimages(3,7,1,22)
count_preimages(5,5,10,-1)
count_preimages(5,5,5,-4)
count_preimages(5,5,2,-25)
count_preimages(5,5,1,-100)
count_preimages(2,2,2,-1)
count_preimages(8,16,16,-1)
count_preimages(18,54,54,-1)
count_preimages(2*4**2,2*4**3,2*4**3,-1)
count_preimages(2*5**2,2*5**3,2*5**3,-1)
count_preimages(2*7**2,2*7**3,2*7**3,-1)
count_preimages(2*11**2,2*11**3,2*11**3,-1)
count_preimages(4,4,4,-3)
count_preimages(36,108,108,-3)
count_preimages(24**2,-649*24**3,5*24**3,16848)
count_preimages(2**4*2**2,3*2**6*2**3,1*2**3,2**12*8)
count_preimages(4*3**4*9**2,5*3**6*9**3,1*9**3,-39*3**12)
count_preimages(4*7**8,5*7**12,7**6,-39*7**12)



