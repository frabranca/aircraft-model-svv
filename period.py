import numpy as np
import Numerical_main as nm
from matplotlib import pyplot as plt
import seaborn as sns
from flightdataprocessing import phugoid_data
from validation_sym import modesym

"""
PERIOD CALCULATION FOR PHUGOID MOTON: this code finds period and amplitude
 for the phugoid motion from the flight data
"""

sns.set_theme()
modesym = modesym(phugoid_data)

tfdlist =[]
qfdlist =[]
tnumlist=[]
qnumlist=[]
for i in range(len(modesym.thplot)):

    if i != 0 and i != len(modesym.thplot)-1:
        if modesym.thplot[i] >= modesym.thplot[i-1] and modesym.thplot[i] >= modesym.thplot[i+1]:
            tfdlist.append(modesym.t[i])
            qfdlist.append(modesym.thplot[i])

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
plt.plot(modesym.t, modesym.thplot, label='flight data')
plt.plot(tfdlist, fit_func(tfdlist, *c_fd))
plt.plot(tnumlist, fit_func(tnumlist, *c_num))

# plt.plot(modesym.t, modesym.y[0].T[0])
plt.plot(tfdlist, qfdlist, 'r.')
plt.plot(tnumlist, qnumlist, 'r.')
# y = fit_func(modesym.t, *popt_fit)
# plt.plot(modesym.t, y)

x0 = 0.1169045
x1 = x0/2
plt.plot(modesym.t, modesym.y[0].T[2], label='numerical')
plt.plot(modesym.t, [x0]*len(modesym.t))
plt.plot(modesym.t, [x1]*len(modesym.t))
plt.legend()


plt.figure()
time = np.linspace(0,1000,len(modesym.t))
plt.plot(time, [x1]*len(modesym.t))
plt.plot(time, fit_func(time, *c_num))
plt.plot(time, fit_func(time, *c_fd))

from scipy.optimize import curve_fit as cv

def fit(x, A, B, C):
    return A*np.exp(B*x)*np.cos(C*x)
c = cv(fit, modesym.t, modesym.qplot)[0]
A, B, C= c
def fit_exp(x, D):
    return A*np.exp(B*x)*np.cos(C*x) + D
c_exp = cv(fit_exp, modesym.t, modesym.qplot)[0]

plt.figure()
plt.plot(modesym.t, fit_exp(modesym.t, *c_exp), 'r')
plt.plot(modesym.t, modesym.qplot, 'k')


def fit_func(x, A, B, angle):
    return A*np.cos(B*x/10 - angle)

popt_fit = cv(fit_func, modesym.t, modesym.uplot)[0]
A, B, angle= popt_fit

def fit_exp(x, A, B, angle, lam):
    return A*np.exp(lam*x)*np.cos( B*x/10 - angle)

popt_exp = cv(fit_exp, modesym.t, modesym.uplot)[0]

plt.figure()
plt.plot(modesym.t, modesym.uplot)
plt.plot(modesym.t, fit(modesym.t, *c))