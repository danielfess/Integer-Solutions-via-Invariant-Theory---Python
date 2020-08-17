from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import os
from os import path
import time

start = time.time()
file = open('sextic_disc.txt')
line = file.readline()
line = file.readline()
disc = parse_expr(line)
middle = time.time()
print('Time to parse:', (middle - start)/60)
factorised = factor(disc)
print(factorised)
end = time.time()
print('Time to factor:',(end - middle)/60)
