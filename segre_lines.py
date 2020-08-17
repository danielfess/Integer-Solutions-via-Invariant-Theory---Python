from sympy import *
import math

f0,f1,f2,f3,f4,f5,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,p12,p13,p14,p15,p23,p24,p25,p34,p35,p45,s,t = symbols('f0 f1 f2 f3 f4 f5 x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 p12 p13 p14 p15 p23 p24 p25 p34 p35 p45 s t')

f1 = 0
f2 = 0
f3 = 0
f4 = 0

z1 = x1*s + y1*t
z2 = x2*s + y2*t
z3 = x3*s + y3*t
z4 = x4*s + y4*t
z5 = x5*s + y5*t

S = expand(f0*f2*z2**3 + f0*f3*z2**2*z3 + f0*f4*z2*z3**2 - f0*f5*z2*z3*z4 + f0*f5*z3**3 - f0*z1*z2**2 + f1*f3*z2**2*z4 + f1*f4*z2*z3*z4 + f1*f5*z3**2*z4 - f1*z2**2*z5 + f2*f4*z2*z4**2 + f2*f5*z3*z4**2 - f2*z1*z2*z4 + f3*f5*z4**3 - f3*z2*z4*z5 - f4*z1*z4**2 - f5*z4**2*z5 + z1**2*z4 - z1*z3*z5 + z2*z5**2)

line_polys = []
for i in range(4):
	poly = S.collect(s).coeff(s,i)
	poly = poly.collect(t).coeff(t,3-i)
	line_polys.append(poly)
	print(poly)

x = [x1,x2,x3,x4,x5]
y = [y1,y2,y3,y4,y5]
p  = dict()
p[(1,2)] = p12
p[(1,3)] = p13
p[(1,4)] = p14
p[(1,5)] = p15
p[(2,3)] = p23
p[(2,4)] = p24
p[(2,5)] = p25
p[(3,4)] = p34
p[(3,5)] = p35
p[(4,5)] = p45
pairs = [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
for (i,j) in pairs:
	poly = p[(i,j)] - x[i-1]*y[j-1] + x[j-1]*y[i-1]
	line_polys.append(poly)
	print(poly)

