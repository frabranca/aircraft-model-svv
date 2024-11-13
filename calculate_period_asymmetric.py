import numpy as np
from flight_data.data_process import approx, dutch_data, spiral_data
import numerical_model as nm
from matplotlib import pyplot as plt
import seaborn as sns
from validation_asymmetric import rplot, y, t

"""
PERIOD CALCULATION FOR DUTCH ROLL MOTION: this code finds period and amplitude
 for the dutch roll motion from the flight data. Two methods are used:
 - computing fitting function
 - finding distance between peaks
"""

sns.set_theme()
# dutch roll
# modeasym = modeasym(dutch_data)
ac = nm.ac()
ac = ac.asym_eig()
P_num = 2*np.pi/np.array([3.814, 2.084, 2.084, -0.009982])
print(P_num)

tfdlist =[]
rfdlist =[]
tnumlist=[]
rnumlist=[]
for i in range(len(rplot)):

    if i != 0 and i != len(rplot)-1:
        if rplot[i] >= rplot[i-1] and rplot[i] >= rplot[i+1]:
            tfdlist.append(t[i])
            rfdlist.append(rplot[i])

    if i != 0 and i != len(y[0].T[2])-1:
        if y[0].T[2][i] >= y[0].T[2][i-1] and y[0].T[2][i] >= y[0].T[2][i+1]:
            tnumlist.append(t[i])
            rnumlist.append(y[0].T[2][i])

tfdlist =np.array([tfdlist[1], tfdlist[2], tfdlist[3]])
rfdlist =np.array([rfdlist[1], rfdlist[2], rfdlist[3]])
tnumlist=np.array([tnumlist[1], tnumlist[2], tnumlist[3]])
rnumlist=np.array([rnumlist[1], rnumlist[2], rnumlist[3]])
#
error = abs(rnumlist - rfdlist)/rfdlist * 100
plt.figure()
plt.plot(tfdlist, rfdlist, 'r.')
plt.plot(tnumlist, rnumlist, 'r.')
plt.plot(t, rplot, label='flight data')
plt.plot(t, y[0].T[2], label='numerical model')
plt.legend()
