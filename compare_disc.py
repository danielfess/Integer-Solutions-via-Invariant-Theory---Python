from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import os
from os import path
import time

start = time.time()

file = open('sextic_disc.txt')
line = file.readline()
line = file.readline()
sextic_disc = parse_expr(line)
middle1 = time.time()
print('Time to parse sextic disc:', (middle1 - start)/60)

file = open('quintic_disc.txt')
line = file.readline()
line = file.readline()
quintic_disc = parse_expr(line)
middle2 = time.time()
print('Time to parse quintic disc:', (middle2 - middle1)/60)

print('Comparison incoming')

if sextic_disc == expand((16*quintic_disc)**3):
	print('Success!!!')
else:
	print('Uh oh....')

end = time.time()
print('Time to compare:',(end - middle2)/60)
