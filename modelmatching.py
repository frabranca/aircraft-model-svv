from Numerical_main import ac, given
import numpy as np
import matplotlib.pyplot as plt
from flightdataprocessing import short_data
import seaborn as sns

cond, tplot, uplot, aplot, thplot, qplot, deplot = short_data
cond = cond.T[0]
x0 = cond[4:8]
# x0 = x0.T[0]

# SHORT PERIOD Cmq =  -0.6
# PHUGOID CXq = not crucialn Cmq used to compensate other coefficients (-3.8)

# Cma = -0.10162
# Cmde = -1.0
# CLa, CD0, e, Cmde, Cma, Cmq, CZq
# Cmq, Cmde, Cma have largest effect
# CZ0 important for phugoid (+0.02)

new    = np.array([5.084, 0.04,      0.8,  -1.1642,  -0.5626, -8.79415, -5.66290])
flight = np.array([6.778, 0.0275, 0.5355, -0.79197, -0.30162, -8.79415, -5.66290])
print(given)
ac1 = ac(m=cond[0], deriv=given, initial=x0, hp0=cond[1], V0=cond[2])
ac2 = ac(m=cond[0], deriv=new, initial=x0, hp0=cond[1], V0=cond[2])

ac2.Cmq    = -7.2
# ac2.Cmadot = -5.
ac2.CZa    = -6.2
ac2.Cma    = -0.3

# ac2.CZ0 = -0.1

# ac2.CXq = +0.01
# ac2.Cmq = -5.79415
# ac2.Cma = -0.5626
# ac2.CZ0 = -0.5
# ac2.CZu = -1.0
# ac2.CZa = -7

time = np.linspace(0., tplot[-1]-tplot[0], len(tplot))
y1 = ac1.sym_input_response(time, deplot, x0)
y2 = ac2.sym_input_response(tplot, deplot, x0)
sns.set_theme()
plt.figure(figsize=(13,9))
f = 20
plt.subplot(221)
plt.xlabel('Time [s]', fontsize=f)
plt.ylabel(r'$\hat{u}$ [-]', fontsize=f)
plt.plot(time, y1[0].T[0], 'b', label='given')
plt.plot(time, y2[0].T[0], 'k:', label='new')
plt.plot(time, uplot,      'r', label='flight data')
plt.subplot(222)
plt.xlabel('Time [s]', fontsize=f)
plt.ylabel(r'$\alpha$ [rad]', fontsize=f)
plt.plot(time, y1[0].T[1], 'b', label='given')
plt.plot(time, y2[0].T[1], 'k:', label='new')
plt.plot(time, aplot,      'r', label='flight data')
plt.subplot(223)
plt.xlabel('Time [s]', fontsize=f)
plt.ylabel(r'$\theta$ [rad]', fontsize=f)
plt.plot(time, y1[0].T[2], 'b', label='given')
plt.plot(time, y2[0].T[2], 'k:', label='new')
plt.plot(time, thplot,      'r', label='flight data')
plt.subplot(224)
plt.xlabel('Time [s]', fontsize=f)
plt.ylabel('qc/V [-]', fontsize=f)
plt.plot(time, y1[0].T[3], 'b', label='given')
plt.plot(time, y2[0].T[3], 'k:', label='new')
plt.plot(time, qplot,      'r', label='flight data')
plt.legend(loc='best', fontsize=15)
plt.show()