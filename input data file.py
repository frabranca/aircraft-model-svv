import pandas as pd
# AIRCRAFT DIMENSION
S = 30. #m^2
c = 2.0569 #m
b = 15.911 #m

# AERODYNAMICS COEFFICIENTS
Cd_0 = 0.04
Cl_a = 5.084
e = 0.8

# AIRCRAFT INERTIA
Kxx2 = 0.019
Kyy2 = 1.3925
Kzz2 = 0.042
Kxz = 0.002

# LONGITUDINAL FORCE DERIVATIVES
CX_u = -0.0279
CX_a = -0.4797
CX_adot = 0.0833
CX_q = -0.2817
CX_d = -0.0373

# NORMAL FORCE DERIVATIVES
CZ_u = -0.3762
CZ_a = -5.7434
CZ_adot = -0.0035
CZ_q = -5.6629
CZ_d = -0.6961

# PITCH MOMENT DERIVATIVES
Cm_u = 0.0699
Cm_a = -0.5626
Cm_adot = 0.1780
Cm_q = -8.7941
Cm_d = -1.1642

Cm_0 = 0.0297
Cm_Tc = -0.0064

# LATERAL FORCE DERIVATIVES
CY_b =
