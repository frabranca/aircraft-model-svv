import numpy as np
from flightdataprocessing import approx, spiral_data
import Numerical_main as nm
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit as cv

"""
Validation code for ASYMMETRIC MOTIONS:
In order to navigate between motions use control+f command to select all instances
of the current motion then change them to a motion of interest. For the nomenclature 
of the motions see bottom of the file "flightdataprocessing. 
"""

def exponent(x,A,l,C):
    return A*np.exp(l/100*x) + C

class modeasym():
    def __init__(self, data):
        self.cond, self.tplot, self.phiplot, self.rplot, self.pplot, self.daplot, self.drplot = data
        self.t = np.linspace(0, (self.tplot[-1] - self.tplot[0]),np.size(self.tplot))
        self.m0 = self.cond[0][0]
        self.h0 = self.cond[1][0]
        self.V0 = self.cond[2][0]
        self.t0 = self.cond[3][0]
        self.beta0 = self.cond[4]
        self.phi0 = self.cond[5][0]
        self.r0 = self.cond[6][0]
        self.p0 = self.cond[7][0]
        self.x0 = np.array([self.beta0, self.phi0, self.r0, self.p0])
        self.totalplot = np.zeros((3,np.size(self.tplot)))
        self.totalplot[0,:] = self.phiplot
        self.totalplot[1,:] = self.rplot
        self.totalplot[2,:] = self.pplot

        # inputs
        self.input = np.zeros((2,np.size(self.tplot)))
        self.input[0,:] = self.daplot
        self.input[1,:] = self.drplot

spiral = modeasym(spiral_data)

spiral_label = ["$Rudder & aileron deflection$", r"$roll angle response$", r"$role rate response$", r"$yaw rate response$"]

ac = nm.ac(m = spiral.m0, initial = spiral.x0, hp0 = spiral.h0, V0 = spiral.V0)
y = ac.asym_input_response(spiral.t, spiral.input.T, spiral.x0)
Af, lf, Cf = cv(exponent, spiral.t, spiral.phiplot)[0]
An, ln, Cn = cv(exponent, spiral.t, y[0].T[1])[0]


sns.set_theme()
plt.figure(figsize=(11,9))
f = 11

plt.title("Yaw rate response", fontsize=f)
plt.ylabel(r"yaw rate [rad]")
plt.xlabel("time [s]")
plt.plot(spiral.t, spiral.phiplot, label='flight data')
plt.plot(spiral.t, exponent(spiral.t, Af, lf, Cf), label="fit on flight data")
plt.plot(spiral.t, exponent(spiral.t, An, ln, Cn), label="fit on numerical simulation")
plt.plot(spiral.t, y[0].T[1], label='numerical model')
plt.legend()
plt.show()

print("flight real eigenvalue", lf/100)
print("numerical real eigenvalue", ln/100)
