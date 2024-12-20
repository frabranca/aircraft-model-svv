import numpy as np
import numerical_model as nm
from matplotlib import pyplot as plt
import seaborn as sns
from flight_data.data_process import phugoid_data
from validation_symmetric import tplot, thplot, uplot, qplot, y, t

"""
PERIOD CALCULATION FOR PHUGOID MOTION: this code finds period and amplitude
 for the phugoid motion from the flight data. Two methods are used:
 - computing fitting function
 - finding distance between peaks 
"""

sns.set_theme()
# modesym = modesym(phugoid_data)

tfdlist =[]
qfdlist =[]
tnumlist=[]
qnumlist=[]
for i in range(len(thplot)):

    if i != 0 and i != len(thplot)-1:
        if thplot[i] >= thplot[i-1] and thplot[i] >= thplot[i+1]:
            tfdlist.append(t[i])
            qfdlist.append(thplot[i])

tfdlist = np.array([tfdlist[1], tfdlist[2], tfdlist[3]])
qfdlist = np.array([qfdlist[1], qfdlist[2], qfdlist[3]])
tnumlist= np.array([tnumlist[4], tnumlist[5], tnumlist[6],])
qnumlist= np.array([qnumlist[4], qnumlist[5], qnumlist[6],])

from scipy.optimize import curve_fit as cv
def fit_func(x, B, C):
    return np.exp(B*x) + C/10

c_fd = cv(fit_func, tfdlist, qfdlist)[0]
c_num = cv(fit_func, tnumlist, qnumlist)[0]
B, C = c_fd

plt.figure()
plt.plot(t, thplot, label='flight data')
plt.plot(tfdlist, fit_func(tfdlist, *c_fd))
plt.plot(tnumlist, fit_func(tnumlist, *c_num))

plt.plot(tfdlist, qfdlist, 'r.')
plt.plot(tnumlist, qnumlist, 'r.')

x0 = 0.1169045
x1 = x0/2
plt.plot(t, y[0].T[2], label='numerical')
plt.plot(t, [x0]*len(t))
plt.plot(t, [x1]*len(t))
plt.legend()


plt.figure()
time = np.linspace(0,1000,len(t))
plt.plot(time, [x1]*len(t))
plt.plot(time, fit_func(time, *c_num))
plt.plot(time, fit_func(time, *c_fd))

from scipy.optimize import curve_fit as cv

def fit(x, A, B, C):
    return A*np.exp(B*x)*np.cos(C*x)
c = cv(fit, t, qplot)[0]
A, B, C= c
def fit_exp(x, D):
    return A*np.exp(B*x)*np.cos(C*x) + D
c_exp = cv(fit_exp, t, qplot)[0]

plt.figure()
plt.plot(t, fit_exp(t, *c_exp), 'r')
plt.plot(t, qplot, 'k')


def fit_func(x, A, B, angle):
    return A*np.cos(B*x/10 - angle)

popt_fit = cv(fit_func, t, uplot)[0]
A, B, angle= popt_fit

def fit_exp(x, A, B, angle, lam):
    return A*np.exp(lam*x)*np.cos( B*x/10 - angle)

popt_exp = cv(fit_exp, t, uplot)[0]

plt.figure()
plt.plot(t, uplot)
plt.plot(t, fit(t, *c))
