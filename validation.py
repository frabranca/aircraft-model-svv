import seaborn as sns
import numpy as np
from flightdataprocessing import short_data, phugoid_data
import Numerical_main as nm
from matplotlib import pyplot as plt

class mode():
    def __init__(self, data):
        self.cond, self.tplot, self.uplot, self.aplot, self.thplot, self.qplot, self.deplot = data
        self.t = np.linspace(0, (self.tplot[-1] - self.tplot[0]) * 60,np.size(self.tplot))
        self.m0 = self.cond[0][0]
        self.h0 = self.cond[1][0]
        self.V0 = self.cond[2][0]
        self.t0 = self.cond[3][0]
        self.u0 = self.cond[4][0]
        self.a0 = np.radians(self.cond[4][0])
        self.th0 = np.radians(self.cond[5][0])
        self.q0 = np.radians(self.cond[6][0])
        self.x0 = np.array([self.u0, self.a0, self.th0, self.q0])

# short = mode(short_data)
phugoid = mode(phugoid_data)
ac_phugoid = nm.ac(m = phugoid.m0, initial = phugoid.x0, hp0 = phugoid.h0, V0 = phugoid.V0)
y = ac_phugoid.sym_input_response(phugoid.t, phugoid.deplot, phugoid.x0)
plt.plot(phugoid.t, y[0])
plt.show()