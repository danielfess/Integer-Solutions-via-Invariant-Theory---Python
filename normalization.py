#Investigating potential normalizations for sextic basis
#Different normalizations differ by a GL_2 covariant.
#For beta_2, beta_3, beta_4, we have the trace-free norm'n,
#and the norm'n for which beta_3^2 = 4 beta_2 beta_4, so
#here we check these two norm'ns differ by a covariant.

from sympy import *

g0 = Symbol('g0')
g1 = Symbol('g1')
g2 = Symbol('g2')
g3 = Symbol('g3')
g4 = Symbol('g4')
g5 = Symbol('g5')

x, y, a, b, c, d = symbols('x, y, a, b, c, d')

x1 = a*x + c*y
y1 = b*x + d*y

h_poly = expand(g0*x1**5 + 5*g1*x1**4*y1 + 10*g2*x1**3*y1**2 + 10*g3*x1**2*y1**3 + 5*g4*x1*y1**4 + g5*y1**5)
h = [0]*6
scale = [1,5,10,10,5,1]
for i in range(6):
	h[i] = (h_poly.coeff(x,5-i)).coeff(y,i)/scale[i]

a2 = h[1]*h[5] - 4*h[2]*h[4] + 3*h[3]**2
a3 = - h[0]*h[5] + 3*h[1]*h[4] - 2*h[2]*h[3]
a4 = h[0]*h[4] - 4*h[1]*h[3] + 3*h[2]**2

print(factor(expand(a2)))
print(factor(expand(a3)))
print(factor(expand(a4)))