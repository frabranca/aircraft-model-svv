import numpy as np
from flightdataprocessing import approx, aperiodic_data
import Numerical_main as nm
from matplotlib import pyplot as plt
import seaborn as sns

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

aperiodic = modeasym(aperiodic_data)

aperiodic_label = ["$Rudder & aileron deflection$", r"$roll angle response$", r"$role rate response$", r"$yaw rate response$"]

ac = nm.ac(m = aperiodic.m0, initial = aperiodic.x0, hp0 = aperiodic.h0, V0 = aperiodic.V0)
y = ac.asym_input_response(aperiodic.t, aperiodic.input.T, aperiodic.x0)

sns.set_theme()
plt.figure(figsize=(11,9))
f = 11

plt.subplot(221)
plt.title("Rudder & aileron deflection", fontsize=f)
plt.ylabel("delfection [rad]")
plt.plot(aperiodic.t, aperiodic.drplot, c ="k", label='rudder')
plt.plot(aperiodic.t, aperiodic.daplot, c="r", label='aileron')
plt.legend()

plt.subplot(222)
plt.title("Roll angle response", fontsize=f)
plt.ylabel("$\phi$ [rad]")
plt.plot(aperiodic.t, aperiodic.phiplot, label='flight data')
plt.plot(aperiodic.t, y[0].T[1], label='numerical model')

plt.subplot(223)
plt.title("Role rate response", fontsize=f)
plt.ylabel(r"roll rate [rad]")
plt.xlabel("time [s]")
plt.plot(aperiodic.t, aperiodic.pplot, label='flight data')
plt.plot(aperiodic.t, y[0].T[2], label='numerical model')

plt.subplot(224)
plt.title("Yaw rate response", fontsize=f)
plt.ylabel(r"yaw rate [rad]")
plt.xlabel("time [s]")
plt.plot(aperiodic.t, aperiodic.rplot, label='flight data')
plt.plot(aperiodic.t, y[0].T[3], label='numerical model')
plt.legend(loc='best')
