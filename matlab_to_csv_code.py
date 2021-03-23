import scipy.io
import numpy as np
import pandas as pd

"""
code used to change the matlabR.mat file in to a CSV format:
unindent line 21 & 22 for use.
"""

mat = scipy.io.loadmat('matlabR.mat')
data = mat["flightdata"][0][0]
labels = np.zeros(len(data),dtype=object)
values = np.zeros(len(data),dtype=object)
for i in range(len(data)):
    labels[i] = data[i][0][0][2][0][0][0]
    values[i] = np.concatenate(data[i][0][0][0])

imax = np.size(values[0])
values = np.concatenate(values).reshape(np.size(values),imax)
df = pd.DataFrame(values.T, columns=labels)
# df.to_csv(r"Flightdata.csv",index=False)
# print(pd.read_csv(r"Flightdata.csv"))
