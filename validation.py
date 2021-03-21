import pandas as pd
import seaborn as sns
import numpy as np
from flightdataprocessing import short_data
from analytical_main import ac
import matplotlib.pyplot as plt
from numpy.linalg import *
import control.matlab as ml

class mode():
    def __init__(self, data):
        self.cond, self.tplot, self.uplot, self.aplot, self.thplot, self.qplot, self.deplot = data
        self.h0 = self.cond[0][0]
        self.V0 = self.cond[1][0]
        self.t0 = self.cond[2][0]
        self.u0 = self.cond[3]
        self.a0 = self.cond[4][0]
        self.th0 = self.cond[5][0]
        self.q0 = self.cond[6][0]
        self.x0 = np.array([self.u0, self.a0, self.th0, self.q0])

short = mode(short_data)
# phugoid = mode(phugoid_data)
ac = ac(hp0=short.h0, V0=short.V0)

t = np.linspace(0,10,len(short.tplot))
y = ac.sym_input_response(t, short.deplot, short.x0)[0]

plt.plot(short.tplot, short.aplot)
plt.plot(short.tplot, y[:,1])
plt.show()
# plt.plot(short.tplot, yshort[0])
# plt.plot(short.tplot, short.uplot)
# plt.show()