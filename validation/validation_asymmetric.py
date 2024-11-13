import numpy as np
from flight_data.data_process import approx, spiral_data
import numerical_model as nm
from matplotlib import pyplot as plt
import seaborn as sns

"""
ASYMMETRIC MOTION VALIDATION (aperiodic roll, dutch roll, dutch roll with yaw dampening, spiral)

This code compares the flight data with the numerical model for the asymmetric eigenmotion.
Change data to to validate the respective motion:
- aperiodic_roll_data;
- dutch_roll_data;
- dutch_roll_yawdamp_data;
- spiral_data ;

"""

data = spiral_data
cond, tplot, phiplot, rplot, pplot, daplot, drplot = data
t = np.linspace(0, (tplot[-1] - tplot[0]),np.size(tplot))

# initial conditions
m0 = cond[0][0]
h0 = cond[1][0]
V0 = cond[2][0]
t0 = cond[3][0]
beta0 = cond[4]
phi0 = cond[5][0]
r0 = cond[6][0]
p0 = cond[7][0]
x0 = np.array([beta0, phi0, r0, p0])

totalplot = np.zeros((3,np.size(tplot)))
totalplot[0,:] = phiplot
totalplot[1,:] = rplot
totalplot[2,:] = pplot

# inputs
input = np.zeros((2,np.size(tplot)))
input[0,:] = daplot
input[1,:] = drplot

spiral_label = ["$Rudder & aileron deflection$", r"$roll angle response$", r"$role rate response$", r"$yaw rate response$"]

ac = nm.ac(m = m0, initial = x0, hp0 = h0, V0 = V0)
y = ac.asym_input_response(t, input.T, x0)

sns.set_theme()
plt.figure(figsize=(11,9))
f = 11

plt.subplot(221)
plt.title("Rudder & aileron deflection", fontsize=f)
plt.ylabel("delfection [rad]")
plt.plot(t, drplot, c ="k", label='rudder')
plt.plot(t, daplot, c="r", label='aileron')
plt.legend()

plt.subplot(222)
plt.title("Roll angle response", fontsize=f)
plt.ylabel("$\phi$ [rad]")
plt.plot(t, phiplot, label='flight data')
plt.plot(t, y[0].T[1], label='numerical model')

plt.subplot(223)
plt.title("Role rate response", fontsize=f)
plt.ylabel(r"p [rad]")
plt.xlabel("time [s]")
plt.plot(t, pplot, label='flight data')
plt.plot(t, y[0].T[2], label='numerical model')

plt.subplot(224)
plt.title("Yaw rate response", fontsize=f)
plt.ylabel(r"r [rad]")
plt.xlabel("time [s]")
plt.plot(t, rplot, label='flight data')
plt.plot(t, y[0].T[3], label='numerical model')
plt.legend(loc='best')

plt.show()
