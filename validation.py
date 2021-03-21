import seaborn as sns
import numpy as np
from flightdataprocessing import short_data, phugoid_data

class mode():
    def __init__(self, data):
        self.cond, self.tplot, self.uplot, self.aplot, self.thplot, self.qplot, self.deplot = data
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