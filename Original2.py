import scipy.integrate as integration
import scipy.misc
from sympy import *
import scipy.optimize as optimize
import numpy


# h=1.054571*10**-14
# m_e=9.1093837*10**-21
# e=1.60217663*10**-9
# epsilon=8.85418*10**-12
# r_0=5.291772*10**-1  #Mass, Length, Time, Charge all scaled by factor of 10^10

h=1
m_e=1
e=1
epsilon=1/(4*pi)
r_0=1

r, a, b = symbols("r a b", real=True)

Psi = E ** (-r / b)

p = -1 / 2 *h**2/m_e* (diff(Psi, r, 2) + 2 / r * diff(Psi, r))

reduced_hamilt = Psi * (p) + Psi ** 2 * (-2*e**2 / (4*pi*epsilon*r))

charge=e* (1 - (1 + 2 * r / a + 2 * r ** 2 / a ** 2) * E ** (-2 * r / a))
potential=integrate(charge/(4*pi*epsilon*r**2) ,r)


def expectation(d1, d2,integrand):
    integra = integrand.subs(b, d2).subs(a, d1)
    def wrapper(x):
        return N(integra.subs(r, x))
    val=integration.quad(wrapper, 0, numpy.inf)
    return val[0]/normalization(d2),val[1]

def normalization(d2):
    norm=Psi**2*r**2
    integra = norm.subs(b, d2)
    def wrapper(x):
        return N(integra.subs(r, x))
    val=integration.quad(wrapper, 0, numpy.inf)
    return val[0]

def total_energy(param):
    x,y=param
    integrand = (reduced_hamilt - e * potential * Psi ** 2) * r ** 2
    first = expectation(r_0 * x, r_0 * y,integrand)[0]
    integrand = (reduced_hamilt) * r ** 2
    second = expectation(1, r_0 * x,integrand)[0]
    return first+second


initial_guess=numpy.array([1,1])
result=optimize.minimize(total_energy,initial_guess)

print(result)
