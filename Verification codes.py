from math import *
import numpy as np
import control.matlab as ml
from sympy import *

hp0 = 5000      	      # pressure altitude in the stationary flight condition [m]
V0 = 100            # true airspeed in the stationary flight condition [m/sec]
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

# unknown eigenvalue
x = symbols('x')
# SYMMETRIC MATRIX
# sym = Matrix([[CXu - 2*muc*x, CXa, CZ0, CXq],
#               [CZu, CZa + (CZadot - 2*muc)*x, -CX0, CZq + 2*muc],
#               [0, 0, -x, 1],
#               [Cmu, Cma + Cmadot*x, 0, Cmq - 2*muc*KY2*x]])

# ASYMMETRIC MATRIX
# asym = Matrix([[CYb + (CYbdot - 2*mub)*x, CL, CYp, CYr - 4*mub],
#                [0, -x/2, 1, 0],
#                [Clb, 0, Clp - 4*mub*KX2*x, Clr + 4*mub*KXZ*x],
#                [Cnb + Cnbdot*x, 0, Cnp + 4*mub*KXZ*x, Cnr - 4*mub*KZ2*x]])

# SIMPLIFICATIONS

# SHORT PERIOD OSCILLATION ----------------------------------------
# ANALYTICAL
short = Matrix([[CZa, CZq + 2*muc],
                [Cma, Cmq]])

eig = Matrix([[(CZadot - 2*muc), 0],
              [Cmadot, -2*muc*KY2]])*x
anal = short + eig
short_eig = np.array(solve(anal.det(),x))*V0/c

real = np.array([re(short_eig[0]), re(short_eig[1])])
imag = np.array([im(short_eig[0]), im(short_eig[1])])

P = 2*pi*c/imag/V0 #period
T12 = np.log(0.5)/real #half amplitude time
mod = (real**2 + imag**2)[0]
freq = sqrt(mod) #frequency
damp = - real/(sqrt(mod)) #damping coefficients

# NUMERICAL
def short_num():
    c11 = [-2 * muc * c / V0, 0, 0, 0]
    c12 = [0, (CZadot - 2 * muc) * c / V0, 0, 0]
    c13 = [0, 0, -c / V0, 0]
    c14 = [0, Cmadot * c / V0, 0, -2 * muc * KY2 * c / V0]
    C1 = np.array([c11,
                   c12,
                   c13,
                   c14])

    c21 = [CXu, CXa, CZ0, CXq]
    c22 = [CZu, CZa, -CX0, CZq + 2 * muc]
    c23 = [0, 0, 0, 1]
    c24 = [Cmu, Cma, 0, Cmq]

    C2 = np.array([c21,
                   c22,
                   c23,
                   c24])

    C3 = np.array([[CXde],
                   [CZde],
                   [0],
                   [Cmde]])

    sa = -np.dot(np.linalg.inv(C1), C2)
    sa[0] = np.zeros((1,4))[0]
    sa[2] = np.zeros((1,4))[0]
    sa[:,0] = np.zeros((4,))
    sa[:,2] = np.zeros((4,))

    sb = -np.dot(np.linalg.inv(C1), C3)
    sc = np.array([[0,0,0,0],
                  [0,1,0,0],
                  [0,0,0,0],
                  [0,0,0,1]])
    sd = np.zeros((4, 1))

    sys = ml.ss(sa, sb, sc, sd)
    return sys, ml.damp(sys, doprint=True)

# s = short_num()[1][2]
# print(s)
# print(short_eig)

# PHUGOID ---------------------------------------------
# ANALYTICAL
phug = Matrix([[CXu - 2*muc*x, CZ0, 0],
               [CZu, 0, 2*muc],
               [0, -x, 1]])

phug_eig = np.array(solve(phug.det(),x))*V0/c

# NUMERICAL
def phug_num():
    c11 = [-2 * muc * c / V0, 0, 0, 0]
    c12 = [0, (CZadot - 2 * muc) * c / V0, 0, 0]
    c13 = [0, 0, -c / V0, 0]
    c14 = [0, Cmadot * c / V0, 0, -2 * muc * KY2 * c / V0]
    C1 = np.array([c11,
                   c12,
                   c13,
                   c14])

    c21 = [CXu, CXa, CZ0, CXq]
    c22 = [CZu, CZa, -CX0, CZq + 2 * muc]
    c23 = [0, 0, 0, 1]
    c24 = [Cmu, Cma, 0, Cmq]

    C2 = np.array([c21,
                   c22,
                   c23,
                   c24])

    C3 = np.array([[CXde],
                   [CZde],
                   [0],
                   [Cmde]])

    sa = -np.dot(np.linalg.inv(C1), C2)
    sa[3] = np.zeros((1,4))[0]
    sa[:,1] = np.zeros((4,))

    sb = -np.dot(np.linalg.inv(C1), C3)
    sc = np.array([[1, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    sd = np.zeros((4, 1))

    sys = ml.ss(sa, sb, sc, sd)
    return sys, ml.damp(sys, doprint=False)

print(phug_num()[0])
# print(phug_eig)
