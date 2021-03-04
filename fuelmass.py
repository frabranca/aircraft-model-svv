import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

d = pd.read_csv('fuelmass.txt', header=None, delimiter= '&')
p = 0.453592
mf = np.hstack((d[0],d[2]))
mf_x = np.hstack((d[1],d[3]))*100
x = mf_x/mf

# INPUT: fuel mass, OUTPUT: x
x_mf = interpolate.interp1d(mf,x,kind='cubic')

# TIME FUNCTION
mflow = -0.048
mstep = 100
tstep = mstep*p/abs(mflow)
t = np.arange(1, len(mf)+1, 1)*tstep

plt.plot(mf, x, 'b.')
# plt.plot(mf, x_mf(mf), 'b')
plt.show()