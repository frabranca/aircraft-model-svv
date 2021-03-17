from math import *
import numpy as np
import control.matlab as ml
from analytical_main import ac
import matplotlib.pyplot as plt
ac = ac()
kts = 0.514444
hp0 = 5000      	      # pressure altitude in the stationary flight condition [m]
V0 = 250*kts            # true airspeed in the stationary flight condition [m/sec]
alpha0 = radians(5)            # angle of attack in the stationary flight condition [rad]
th0 = radians(4)            # pitch angle in the stationary flight condition [rad]
rho0, Temp0, R = 1.2250, 288.15, 287.05          # air density, temperature at sea level [kg/m^3, K] + GAS CONSTANT
g = 9.81
W = 60500           # [N]       (aircraft weight)
m = W/g
dt = 0.01
t = np.arange(0., 20.+dt, dt)
# Aerodynamic properties
e = 0.8            # Oswald factor [ ]
CD0 = 0.04            # Zero lift drag coefficient [ ]
CLa = 5.084            # Slope of CL-alpha curve [ ]
# Longitudinal stability
Cma = -0.5626            # longitudinal stabilty [ ]
Cmde = -1.1642            # elevator effectiveness [ ]

# Aircraft geometry
S = 30.	          # wing area [m^2]
Sh = 0.2 * S         # stabiliser area [m^2]
Sh_S = Sh / S	          # [ ]
lh = 0.71 * 5.968    # tail length [m]
c = 2.0569	          # mean aerodynamic cord [m]
lh_c = lh / c	          # [ ]
b = 15.911	          # wing span [m]
bh = 5.791	          # stabiliser span [m]
A = b ** 2 / S      # wing aspect ratio [ ]
Ah = bh ** 2 / Sh    # stabiliser aspect ratio [ ]
Vh_V = 1	          # [ ]
ih = -2 * pi / 180   # stabiliser angle of incidence [rad]

lam = -0.0065         # temperature gradient in ISA [K/m]
rho = rho0 * pow( ((1+(lam * hp0 / Temp0))), (-((g / (lam * R)) + 1)))

# muc = m / (rho * S * c)
muc = m / (rho * S * c)
mub = m / (rho * S * b)

KX2, KZ2, KXZ, KY2 = 0.019, 0.042, 0.002, 1.25 * 1.114

# Aerodynamic constants

Cmac = 0                      # Moment coefficient about the aerodynamic centre [ ]
CNwa = CLa                    # Wing normal force slope [ ]
CNha = 2 * pi * Ah / (Ah + 2) # Stabiliser normal force slope [ ]
depsda = 4 / (A + 2)            # Downwash gradient [ ]

# Lift and drag coefficient

CL = 2 * W / (rho * V0 ** 2 * S)              # Lift coefficient [ ]
CD = CD0 + (CLa * alpha0) ** 2 / (pi * A * e) # Drag coefficient [ ]

# Stability derivatives

CX0 = W * sin(th0) / (0.5 * rho * V0 ** 2 * S)
CXu,CXa,CXadot,CXq,CXde = -0.0279, +0.47966, +0.08330, -0.28170, -0.03728

CZ0 = -W * cos(th0) / (0.5 * rho * V0 ** 2 * S)
CZu, CZa, CZadot, CZq, CZde = -0.37616, -5.74340, -0.00350, -5.66290, -0.69612

Cmu, Cmadot, Cmq = +0.06990, +0.17800, -8.79415

CYb = -0.7500
CYbdot =  0
CYp = -0.0304
CYr = +0.8495
CYda = -0.0400
CYdr = +0.2300

Clb = -0.10260
Clp = -0.71085
Clr = +0.23760
Clda = -0.23088
Cldr = +0.03440

Cnb =  +0.1348
Cnbdot = 0
Cnp = -0.0602
Cnr = -0.2061
Cnda = -0.0120
Cndr = -0.0939

# NUMERICAL MODEL----------------------------------------------
sym = ac.sym_system(V0)
symfunc = ml.damp(sym, doprint=False)
sym_freq = symfunc[0]
sym_damp = symfunc[1]
sym_eig = symfunc[2]
    # manoeuvres
short_num = sym_eig[0:2] # SHORT PERIOD
phug_num = sym_eig[2:4] # PHUGOID

asym = ac.asym_system(V0)
asymfunc = ml.damp(asym, doprint=False)
asym_freq = asymfunc[0]
asym_damp = asymfunc[1]
asym_eig = asymfunc[2]

#ANALYTICAL MODEL-----------------------------------------------
from sympy import *
x = symbols('x')
# SHORT PERIOD CZadot = 0, CZq = 0
short = Matrix([[CZa - 2*muc*x, 2*muc],
                [Cma + Cmadot*x, Cmq - 2*muc*KY2*x]])
short = short.det()
short_eig = np.array(solve(short,x))*V0/c
eshort = abs(np.abs(short_eig[0]) - np.abs(sym_eig[0])) / np.abs(short_eig[0])*100

# PHUGOID
phug = Matrix([[CXu - 2*muc*x, CXa, CZ0, CXq],
               [CZu, CZa, 0, 2*muc],
               [0, 0, -x, 1],
               [Cmu, Cma, 0, Cmq]])
# phug = Matrix([[CXu - 2*muc*x, CXa, CZ0, CXq],
#                [CZu, CZa, 0, 2*muc],
#                [0, 0, -x, 1],
#                [Cmu, Cma, 0, Cmq]])
phug = phug.det()
phug_eig = np.array(solve(phug,x))*V0/c
ephug = abs(np.abs(phug_eig[0]) - np.abs(sym_eig[0])) # / np.abs(phug_eig[0])*100

# a = -4*muc**2
# b = 2*muc*CXu
# c = -CZu*CZ0
# peig = np.roots([a,b,c])*V0/c

# APERIODIC ROLL
ap_eig = Clp / (4*mub*KX2) *V0/b
eap = abs(np.abs(ap_eig) - np.abs(asym_eig[0]))/ np.abs(ap_eig)*100

# SPIRAL
n = 2*CL*(Clb*Cnr - Cnb*Clr)
d = Clp*(CYb*Cnr + 4*mub*Cnb) - Cnp*(CYb*Clr + 4*mub*Clb)
sp_eig = n/d*V0/b

# DUTCH ROLL + APERIODIC ROLL
dutch = Matrix([[-Clb + .5*Clr*x + 2*mub*KXZ*x**2, Clp - 4*mub*KX2*x],
                [-Cnb + .5*Cnr*x - 2*mub*KZ2*x**2, Cnp + 4*mub*KXZ*x]])
dutch = dutch.det()
dutch_eig = np.array(solve(dutch,x))*V0/b