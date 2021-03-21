import numpy as np
from flightdataprocessing import phugoid_data, approx
import Numerical_main as nm
from matplotlib import pyplot as plt



class mode():
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
phugoid = mode(phugoid_data)

tapprox, deapprox = approx(phugoid.tplot, phugoid.deplot, step=2)
print(deapprox)

ac_phugoid = nm.ac(m = phugoid.m0, initial = phugoid.x0, hp0 = phugoid.h0, V0 = phugoid.V0)
y = ac_phugoid.sym_input_response(phugoid.t, phugoid.deplot, phugoid.x0)
plt.subplot(121)
plt.title("numerical model")
# plt.plot(phugoid.t, y[0].T[0], label="u")
# plt.plot(phugoid.t, y[0].T[1], label="alpha")
# plt.plot(phugoid.t, y[0].T[2], label="theta")
# plt.plot(phugoid.t, phugoid.qplot)
# plt.plot(phugoid.t, y[0].T[3], label="q")
plt.plot(phugoid.t, phugoid.deplot)
plt.legend()
plt.subplot(122)
plt.plot(deapprox, deapprox)
# plt.title("flight data")

plt.show()