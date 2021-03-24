import numpy as np
from flightdataprocessing import approx, dutch_data, spiral_data
import Numerical_main as nm
from matplotlib import pyplot as plt
import seaborn as sns
from validationfinal import modeasym

"""
PERIOD CALCULATION FOR DUTCH ROLL MOTION: this code finds period and amplitude
 for the dutch roll motion from the flight data. Two methods are used:
 - computing fitting function
 - finding distance between peaks
"""

sns.set_theme()
# dutch roll
modeasym = modeasym(dutch_data)
ac = nm.ac()
ac = ac.asym_eig()
P_num = 2*np.pi/np.array([3.814, 2.084, 2.084, -0.009982])
print(P_num)

modeasym.plot()

tfdlist =[]
rfdlist =[]
tnumlist=[]
rnumlist=[]
for i in range(len(modeasym.rplot)):

    if i != 0 and i != len(modeasym.rplot)-1:
        if modeasym.rplot[i] >= modeasym.rplot[i-1] and modeasym.rplot[i] >= modeasym.rplot[i+1]:
            tfdlist.append(modeasym.t[i])
            rfdlist.append(modeasym.rplot[i])

    if i != 0 and i != len(modeasym.y[0].T[2])-1:
        if modeasym.y[0].T[2][i] >= modeasym.y[0].T[2][i-1] and modeasym.y[0].T[2][i] >= modeasym.y[0].T[2][i+1]:
            tnumlist.append(modeasym.t[i])
            rnumlist.append(modeasym.y[0].T[2][i])

tfdlist =np.array([tfdlist[1], tfdlist[2], tfdlist[3]])
rfdlist =np.array([rfdlist[1], rfdlist[2], rfdlist[3]])
tnumlist=np.array([tnumlist[1], tnumlist[2], tnumlist[3]])
rnumlist=np.array([rnumlist[1], rnumlist[2], rnumlist[3]])
#
error = abs(rnumlist - rfdlist)/rfdlist * 100
plt.figure()
plt.plot(tfdlist, rfdlist, 'r.')
plt.plot(tnumlist, rnumlist, 'r.')
plt.plot(modeasym.t, modeasym.rplot, label='flight data')
plt.plot(modeasym.t, modeasym.y[0].T[2], label='numerical model')
plt.legend()