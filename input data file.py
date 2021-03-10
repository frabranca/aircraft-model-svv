class d:
    def __init__(self):
        self.e = 0.8            # Oswald factor [ ]
        self.CD0 = 0.04            # Zero lift drag coefficient [ ]
        self.CLa = 5.084            # Slope of CL-alpha curve [ ]
        # Longitudinal stability
        self.Cma = -0.5626            # longitudinal stabilty [ ]
        self.Cmde = -1.1642            # elevator effectiveness [ ]


import scipy.io
import numpy as np
import pandas as pd
mat = scipy.io.loadmat('matlabR.mat')
# print(mat)
data = mat['flightdata']
files = data[0][0]

aoa = files[0][0][0][0]
dte = files[1][0][0][0]
time = files[48][0][0][0]


