import analytical_main as am
import numpy as np
import control.matlab as ml
from matplotlib import pyplot as plt

def T(lam):
    return np.log(0.5)/np.real(lam)
def P(lam):
    return 2*np.pi/np.imag(lam)
def Co(lam):
    return -np.log(0.5)/(2*np.pi)*(np.imag(lam)/np.real(lam))
def delta(lam):
    return 2*np.pi*(np.real(lam)/np.imag(lam))
def damp(lam):
    return -np.real(lam)/(np.sqrt(np.real(lam)**2 + np.imag(lam)**2))

"""
solve equations of motion for inital alpha and velocity
"""

if __name__ == "__main__":

    ac = am.ac(hp0=5000)
    V = 100
    sys1 = ac.sym_system(V)
    poles, eigenvectors = np.linalg.eig(ac.A)
    array = ml.damp(sys1)
    print(delta(array[2]))


