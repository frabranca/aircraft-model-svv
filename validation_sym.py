import numpy as np
from flightdataprocessing import approx, phugoid_data
import Numerical_main as nm
from matplotlib import pyplot as plt
import seaborn as sns

class modesym():
    def __init__(self, data):
        self.cond, self.tplot, self.uplot, self.aplot, self.thplot, self.qplot, self.deplot = data
        self.t = np.linspace(0, (self.tplot[-1] - self.tplot[0]),np.size(self.tplot))
        self.m0 = self.cond[0][0]
        self.h0 = self.cond[1][0]
        self.V0 = self.cond[2][0]
        self.t0 = self.cond[3][0]
        self.u0 = self.cond[4][0]
        self.a0 = self.cond[5][0]
        self.th0 = self.cond[6][0]
        self.q0 = self.cond[7][0]
        self.x0 = np.array([self.u0, self.a0, self.th0, self.q0])
        self.totalplot = np.zeros((4,np.size(self.tplot)))
        self.totalplot[0,:] = self.uplot
        self.totalplot[1,:] = self.aplot
        self.totalplot[2,:] = self.thplot
        self.totalplot[3,:] = self.qplot

phugoid = modesym(phugoid_data)

phugoid_label = ["$Rudder & aileron deflection$", r"$roll angle response$", r"$role rate response$", r"$yaw rate response$"]

ac = nm.ac(m = phugoid.m0, initial = phugoid.x0, hp0 = phugoid.h0, V0 = phugoid.V0)
y = ac.sym_input_response(phugoid.t, phugoid.deplot, phugoid.x0)

sns.set_theme()
plt.figure(figsize=(11,9))
f = 11

plt.subplot(221)
plt.title("Unitless velocity response", fontsize=f)
plt.ylabel("$\hat{u}$ [-]")
plt.plot(phugoid.t, phugoid.uplot, label='flight data')
plt.plot(phugoid.t, y[0].T[0], label='numerical model')

plt.subplot(222)
plt.title("Angle of attack response", fontsize=f)
plt.ylabel(r"$\Delta \alpha$ [rad]")
plt.plot(phugoid.t, phugoid.aplot, label='flight data')
plt.plot(phugoid.t, y[0].T[1], label='numerical model')

plt.subplot(223)
plt.title("Flight angle response", fontsize=f)
plt.ylabel(r"$\Delta \theta$ [rad]")
plt.xlabel("time [s]")
plt.plot(phugoid.t, phugoid.thplot, label='flight data')
plt.plot(phugoid.t, y[0].T[2], label='numerical model')

plt.subplot(224)
plt.title("Pitch rate response", fontsize=f)
plt.ylabel(r"pitch rate [rad]")
plt.xlabel("time [s]")
plt.plot(phugoid.t, phugoid.qplot, label='flight data')
plt.plot(phugoid.t, y[0].T[3], label='numerical model')
plt.legend(loc='best')
plt.show()
plt.title("Elevator deflection")
plt.plot(phugoid.t, phugoid.deplot, c="k")
plt.ylabel("deflection [rad]")
plt.xlabel("time")
plt.show()