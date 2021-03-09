from math import *
import numpy as np
from numpy.linalg import *
import control.matlab as ml
import matplotlib.pyplot as plt


class ac:
    def __init__(self, hp0=8000):
        # Citation 550 - Linear simulation
        # xcg = 0.25 * c
        # Stationary flight condition

        self.hp0 = hp0  # pressure altitude in the stationary flight condition [m]
        self.V0 = 102.888  # true airspeed in the stationary flight condition [m/sec]
        self.alpha0 = np.radians(5)  # angle of attack in the stationary flight condition [rad]
        self.th0 = np.radians(4)  # pitch angle in the stationary flight condition [rad]
        self.rho0 = 1.2250  # air density at sea level [kg/m^3]
        self.Temp0 = 288.15  # temperature at sea level in ISA [K]
        self.R = 287.05  # specific gas constant [m^2/sec^2K]
        self.g = 9.81
        self.W = 60500  # [N]       (aircraft weight)
        self.m = self.W / self.g

        # Aerodynamic properties
        self.e = 0.8  # Oswald factor [ ]
        self.CD0 = 0.04  # Zero lift drag coefficient [ ]
        self.CLa = 5.084  # Slope of CL-alpha curve [ ]
        # Longitudinal stability
        self.Cma = -0.5626  # longitudinal stabilty [ ]
        self.Cmde = -1.1642  # elevator effectiveness [ ]

        # Aircraft geometry

        self.S = 30.00  # wing area [m^2]
        self.Sh = 0.2 * self.S  # stabiliser area [m^2]
        self.Sh_S = self.Sh / self.S  # [ ]
        self.lh = 0.71 * 5.968  # tail length [m]
        self.c = 2.0569  # mean aerodynamic cord [m]
        self.lh_c = self.lh / self.c  # [ ]
        self.b = 15.911  # wing span [m]
        self.bh = 5.791  # stabiliser span [m]
        self.A = self.b ** 2 / self.S  # wing aspect ratio [ ]
        self.Ah = self.bh ** 2 / self.Sh  # stabiliser aspect ratio [ ]
        self.Vh_V = 1  # [ ]
        self.ih = -2 * pi / 180  # stabiliser angle of incidence [rad]

        self.lam = -0.0065  # temperature gradient in ISA [K/m]
        self.rho = self.rho0 * pow(((1 + (self.lam * self.hp0 / self.Temp0))), (-((self.g / (self.lam * self.R)) + 1)))
        self.muc = self.m / (self.rho * self.S * self.c)
        self.mub = self.m / (self.rho * self.S * self.b)

        self.KX2 = 0.019
        self.KZ2 = 0.042
        self.KXZ = 0.002
        self.KY2 = 1.25 * 1.114

        # Aerodynamic constants

        self.Cmac = 0  # Moment coefficient about the aerodynamic centre [ ]
        self.CNwa = self.CLa  # Wing normal force slope [ ]
        self.CNha = 2 * pi * self.Ah / (self.Ah + 2)  # Stabiliser normal force slope [ ]
        self.depsda = 4 / (self.A + 2)  # Downwash gradient [ ]

        # Lift and drag coefficient

        self.CL = 2 * self.W / (self.rho * self.V0 ** 2 * self.S)  # Lift coefficient [ ]
        self.CD = self.CD0 + (self.CLa * self.alpha0) ** 2 / (pi * self.A * self.e)  # Drag coefficient [ ]

        # Stability derivatives

        self.CX0 = self.W * sin(self.th0) / (0.5 * self.rho * self.V0 ** 2 * self.S)
        self.CXu = -0.0279
        self.CXa = +0.47966  # Positive! (see FD lecture notes)
        self.CXadot = +0.08330
        self.CXq = -0.28170
        self.CXde = -0.03728

        self.CZ0 = -self.W * cos(self.th0) / (0.5 * self.rho * self.V0 ** 2 * self.S)
        self.CZu = -0.37616
        self.CZa = -5.74340
        self.CZadot = -0.00350
        self.CZq = -5.66290
        self.CZde = -0.69612

        self.Cmu = +0.06990
        self.Cmadot = +0.17800
        self.Cmq = -8.79415

        self.CYb = -0.7500
        self.CYbdot = 0
        self.CYp = -0.0304
        self.CYr = +0.8495
        self.CYda = -0.0400
        self.CYdr = +0.2300

        self.Clb = -0.10260
        self.Clp = -0.71085
        self.Clr = +0.23760
        self.Clda = -0.23088
        self.Cldr = +0.03440

        self.Cnb = +0.1348
        self.Cnbdot = 0
        self.Cnp = -0.0602
        self.Cnr = -0.2061
        self.Cnda = -0.0120
        self.Cndr = -0.0939

    # SYMMETRIC (C1, C2, C3)
    def sym_system(self, V):
        c11 = [-2 * self.muc * self.c / V, 0, 0, 0]
        c12 = [0, (self.CZadot - 2 * self.muc) * self.c / V, 0, 0]
        c13 = [0, 0, -self.c / V, 0]
        c14 = [0, self.Cmadot * self.c / V, 0, -2 * self.muc * self.KY2 * self.c / V]
        C1 = np.array([c11,
                       c12,
                       c13,
                       c14])

        c21 = [self.CXu, self.CXa, self.CZ0, self.CXq]
        c22 = [self.CZu, self.CZa, -self.CX0, self.CZq + 2 * self.muc]
        c23 = [0, 0, 0, 1]
        c24 = [self.Cmu, self.Cma, 0, self.Cmq]

        C2 = np.array([c21,
                       c22,
                       c23,
                       c24])

        C3 = np.array([[self.CXde],
                       [self.CZde],
                       [0],
                       [self.Cmde]])

        self.A = -np.matmul(np.linalg.inv(C1), C2)
        self.B = -np.matmul(np.linalg.inv(C1), C3)
        self.C = np.eye(4)
        self.D = np.zeros((4, 1))

        self.sys = ml.ss(self.A, self.B, self.C, self.D)

        return self.sys

    def sym_response(self, V):
        system = self.sym_system(V)
        return system.A

    def asym(self, V):
        c11 = [(self.CYbdot - 2 * self.mub) * self.b / V, 0, 0, 0]
        c12 = [0, -self.b / (2 * V), 0, 0]
        c13 = [0, 0, -2 * self.mub * self.KX2 * (self.b / V)**2, 2 * self.mub * self.KXZ * (self.b / V)**2]
        c14 = [self.Cnbdot * self.b / V, 0, 2 * self.mub * self.KXZ * (self.b / V)**2,
               -2 * self.mub * self.KZ2 * (self.b / V)**2]
        C1 = np.array([c11,
                       c12,
                       c13,
                       c14])

        c21 = [self.CYb, self.CL, self.CYp*self.b/(2*V), (self.CYr - 4 * self.mub)*self.b/(2*V)]
        c22 = [0, 0, self.b/(2*V), 0]
        c23 = [self.Clb, 0, self.Clp*self.b/(2*V), self.Clr*self.b/(2*V)]
        c24 = [self.Cnb, 0, self.Cnp*self.b/(2*V), self.Cnr*self.b/(2*V)]
        C2 = np.array([c21,
                       c22,
                       c23,
                       c24])

        C3 = np.array([[self.CYda, self.CYdr],
                       [0, 0],
                       [self.Clda, self.Cldr],
                       [self.Cnda, self.Cndr]])

        self.A =  -np.matmul(np.linalg.inv(C1), C2)
        self.B = -np.matmul(np.linalg.inv(C1), C3)
        self.C = np.eye(4)
        self.D = np.zeros((4, 2))

        self.sys = ml.ss(self.A, self.B, self.C, self.D)
        return self.sys

