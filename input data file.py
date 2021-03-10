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


