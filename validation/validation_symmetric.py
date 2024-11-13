import numpy as np
from flight_data.data_process import approx, short_period_data, phugoid_data
import numerical_model as nm
from matplotlib import pyplot as plt
import seaborn as sns

"""
SYMMETRIC MOTION VALIDATION (short period and phugoid)

This code compares the flight data with the numerical model for the symmetric eigenmotion.
Change data to validate the respective motion:
- short_period_data;
- phugoid_data;

"""

data = short_period_data
cond, tplot, uplot, aplot, thplot, qplot, deplot = data
t = np.linspace(0, (tplot[-1] - tplot[0]),np.size(tplot))

# initial conditions
m0 = cond[0][0]
h0 = cond[1][0]
V0 = cond[2][0]
t0 = cond[3][0]
u0 = cond[4][0]
a0 = cond[5][0]
th0 = cond[6][0]
q0 = cond[7][0]
x0 = np.array([u0, a0, th0, q0])

totalplot = np.zeros((4,np.size(tplot)))
totalplot[0,:] = uplot
totalplot[1,:] = aplot
totalplot[2,:] = thplot
totalplot[3,:] = qplot

short_label = ["$Rudder & aileron deflection$", r"$roll angle response$", r"$role rate response$", r"$yaw rate response$"]

ac = nm.ac(m = m0, initial = x0, hp0 = h0, V0 = V0)
y = ac.sym_input_response(t, deplot, x0)

sns.set_theme()
plt.figure(figsize=(11,9))
f = 11

plt.subplot(221)
plt.title("Unitless velocity response", fontsize=f)
plt.ylabel("$\hat{u}$ [-]")
plt.plot(t, uplot, label='flight data')
plt.plot(t, y[0].T[0], label='numerical model')

plt.subplot(222)
plt.title("Angle of attack response", fontsize=f)
plt.ylabel(r"$\Delta \alpha$ [rad]")
plt.plot(t, aplot, label='flight data')
plt.plot(t, y[0].T[1], label='numerical model')

plt.subplot(223)
plt.title("Flight angle response", fontsize=f)
plt.ylabel(r"$\Delta \theta$ [rad]")
plt.xlabel("time [s]")
plt.plot(t, thplot, label='flight data')
plt.plot(t, y[0].T[2], label='numerical model')

plt.subplot(224)
plt.title("Pitch rate response", fontsize=f)
plt.ylabel(r"q [rad]")
plt.xlabel("time [s]")
plt.plot(t, qplot, label='flight data')
plt.plot(t, y[0].T[3], label='numerical model')
plt.legend(loc='best')
plt.show()
plt.title("Elevator deflection")
plt.plot(t, deplot, c="k")
plt.ylabel("deflection [rad]")
plt.xlabel("time [s]")

plt.show()