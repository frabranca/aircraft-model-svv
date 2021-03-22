import numpy as np
from flightdataprocessing import approx, search_data
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

search = modeasym(search_data)

search_label = ["$deflection$", r"$\phi$", r"$\frac{pb}{2V}$", r"$\frac{rb}{2V}$"]

ac = nm.ac(m = search.m0, initial = search.x0, hp0 = search.h0, V0 = search.V0)
y = ac.asym_input_response(search.t, search.input.T, search.x0)

sns.set_theme()
plt.figure(figsize=(11,9))
f = 20

plt.subplot(221)
plt.title(search_label[0], fontsize=f)
plt.plot(search.t, search.daplot, c="r", label='aileron')
plt.plot(search.t, search.drplot, c ="k", label='ruddor')
plt.legend()

plt.subplot(222)
plt.title(search_label[1], fontsize=f)
plt.plot(search.t, y[0].T[1], label='numerical model')
plt.plot(search.t, search.phiplot, label='flight data')

plt.subplot(223)
plt.title(search_label[2], fontsize=f)
plt.plot(search.t, y[0].T[2], label='numerical model')
plt.plot(search.t, search.pplot, label='flight data')

plt.subplot(224)
plt.title(search_label[3], fontsize=f)
plt.plot(search.t, y[0].T[3], label='numerical model')
plt.plot(search.t, search.rplot, label='flight data')
plt.legend(loc='best')
