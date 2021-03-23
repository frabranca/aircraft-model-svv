import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

mat = scipy.io.loadmat('matlabR.mat')
# print(mat)
data = mat['flightdata']
files = data[0][0]

aoa = files[0][0][0][0].T[0]
dte = files[1][0][0][0].T[0]
time = files[48][0][0][0][0]

# plt.plot(time, dte)
# plt.show()

col = []
for i in range(len(files)):
    col.append(files[i][0][0][2][0][0][0])
# val = []
# for i in range(len(files)):
#     val.append(files[i][0][0][0])



with open('dat.txt') as csv_file:
    csv_reader = csv.writer(csv_file)
    writer = csv.DictWriter(csv_file, fieldnames=col)
    writer.writerow({'Angle of attack': 'John Smith'})
    # writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})