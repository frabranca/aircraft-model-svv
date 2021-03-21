import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

fd = pd.read_csv('data.csv')
t = fd['t'].values

# SYMMETRIC STATE VARIABLES
tas = fd['tas'].values * 0.514444      # True airspeed
a = np.radians(fd['a'].values)          # angle of attack [rad]
theta = np.radians(fd['theta'].values)  # pitch angle [rad]
q = np.radians(fd['q'].values)          # pitch rate [rad/s]

# SYMMETRIC INPUTS [deg]
de = np.radians(fd['de'].values)        # elevator deflection [rad]

# ASYMMETRIC STATE VARIABLES
phi = np.radians(fd['phi'].values)      # roll angle [rad]
p = np.radians(fd['p'].values)          # roll rate [rad]
r = np.radians(fd['r'].values)          # yaw rate [rad]

# ASYMMETRIC INPUTS [deg]
da = np.radians(fd['da'].values)        # aileron deflection [rad]
dr = np.radians(fd['dr'].values)        # rudder deflection [rad]

# ENVIRONMENT
hp = fd['hp'].values * 0.3048
h = fd['h correct'].values * 0.3048
cas = fd['cas'].values * 0.514444

# mass
kg = 0.453592
minitial = 6072.8642115 #kg
fu = (fd['fu1'].values + fd['fu2'].values)*kg
m = minitial - fu

# MANOEUVRES TIME INTERVALS
short = (3685, 3685 + 20)
phugoid = (55*60 + 44, 58*60 + 20)
dutch = (59*60+2, 59*60 + 25 + 5 + 2)
dutch_yawdamp = (59*60 + 50 + 5, 59*60 + 70 + 5)
aperiodic = (60*60 + 2*60 + 31+8, 60*60 + 2*60 + 61+8)
spiral = (60*60 + 4*60, 4300)

def approx(t, x, step=35):
    xx = np.array(x)
    tt = np.array(t)

    n = np.size(tt)%step
    xx = xx[0:-n]
    tt = tt[0:-n]

    xx = xx.reshape(int(np.size(xx)/step), step)
    tt = tt.reshape(int(np.size(tt)/step), step)

    xx = np.average(xx, axis=1)
    tt = np.average(tt, axis=1)

    return tt, xx

def symplot(mode):
    start = mode[0]
    end = mode[1]

    # time
    tplot = t[(t > start) & (t < end)] #s
    # velocity
    vt = tas[(t > start) & (t < end)] #m/s

    # initial conditions
    c = 2.0569
    ind = np.where(t == tplot[0])[0]-1
    m0 = m[ind] # kg
    h0 = hp[ind] # m
    V0 = tas[ind] #m/s
    t0 = t[ind] #s
    u0 = (vt[0]-V0)/V0 # -
    a0 = a[ind] #rad
    th0 = theta[ind] #rad
    q0 = q[ind]*c/V0 #rad/s

    # state variables
    uplot = (vt - V0)/V0 # -
    aplot = a[(t > start) & (t < end)]
    thplot = theta[(t > start) & (t < end)]
    qplot = q[(t > start) & (t < end)] * c/V0
    deplot = de[(t > start) & (t < end)]

    tplot = tplot/60
    f=20

    # plt.figure(figsize=(12,8))
    # tt, uu = approx(tplot, uplot)
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Normalized velocity change [-]', fontsize=f)
    # #     plt.plot(tt, uu, 'k')
    # plt.plot(tplot, uplot, 'c')
    #
    # tt, aa = approx(tplot, aplot)
    # plt.figure(figsize=(12,8))
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Angle of attack [rad]', fontsize=f)
    # plt.plot(tt, aa, 'k', label='Approximated')
    # plt.plot(tplot, aplot, 'c', label='Real')
    # plt.legend(fontsize=f)
    #
    # tt, thth = approx(tplot, thplot)
    # plt.figure(figsize=(12,8))
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Pitch angle [rad]', fontsize=f)
    # #     plt.plot(tt, thth, 'k')
    # plt.plot(tplot, thplot, 'c')
    #
    # tt, qq = approx(tplot, qplot)
    # plt.figure(figsize=(12,8))
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Pitch rate [rad/s]', fontsize=f)
    # plt.plot(tt, qq, 'k', label='Approximated')
    # plt.plot(tplot, qplot, 'c', label='Real')
    # plt.legend(fontsize=f)
    #
    #
    # tt, dede = approx(tplot, deplot)
    # plt.figure(figsize=(12,8))
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Elevator deflection [rad]', fontsize=f)
    # plt.plot(tt, dede, 'k', label='Approximated')
    # plt.plot(tplot, deplot, 'c', label='Real')
    # plt.legend(fontsize=f)

    x0 = np.array([m0, h0, V0, t0, u0, a0, th0, q0])
    return x0, tplot, uplot, aplot, thplot, qplot, deplot

# symplot(phugoid)

def asymplot(mode):
    sns.set_theme()
    start = mode[0]
    end = mode[1]

    # time
    tplot = t[(t > start) & (t < end)] # s
    mplot = m[(t > start) & (t < end)] # kg

    # initial conditions
    b = 15.911
    ind = np.where(t == tplot[0])[0]-1
    m0 = mplot[ind] # kg
    h0 = hp[ind] # m
    V0 = tas[ind] # m/s
    t0 = t[ind] # s
    beta0 = 0.
    phi0 = phi[ind]
    p0 = p[ind]*b/V0/2
    r0 = r[ind]*b/V0/2

    tplot = tplot/60
    f=20
    # state variables
    #     bplot =
    phiplot = phi[(t > start) & (t < end)]
    pplot = p[(t > start) & (t < end)] *b/V0/2
    rplot = r[(t > start) & (t < end)] *b/V0/2
    daplot = da[(t > start) & (t < end)]
    drplot = dr[(t > start) & (t < end)]

    # roll angle
    # plt.figure(figsize=(12,8))
    # tt, phi_approx = approx(tplot, phiplot)
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Roll angle [rad]', fontsize=f)
    # plt.plot(tplot, phiplot, 'c')
    # plt.plot(tt, phi_approx, 'k')
    #
    # # roll rate
    # tt, p_approx = approx(tplot, pplot, step = 50)
    # plt.figure(figsize=(12,8))
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Roll rate', fontsize=f)
    # plt.plot(tplot, pplot, 'c')
    # plt.plot(tt, p_approx, 'k')
    #
    # # yaw rate
    # tt, r_approx = approx(tplot, rplot)
    # plt.figure(figsize=(12,8))
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Yaw rate', fontsize=f)
    # plt.plot(tplot, rplot, 'c')
    # plt.plot(tt, r_approx, 'k')
    #
    # # aileron deflection
    # tt, da_approx = approx(tplot, daplot)
    # plt.figure(figsize=(12,8))
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Aileron deflection', fontsize=f)
    # plt.plot(tplot, daplot, 'c')
    # plt.plot(tt, da_approx, 'k')
    #
    # # rudder deflection
    # tt, dr_approx = approx(tplot, drplot)
    # plt.figure(figsize=(12,8))
    # plt.xlabel('Time [min]', fontsize=f)
    # plt.ylabel('Rudder deflection', fontsize=f)
    # plt.plot(tplot, drplot, 'c')
    # plt.plot(tt, dr_approx, 'k')


    x0 = np.array([m0, h0, V0, t0, beta0, phi0, p0, r0])
    return x0, tplot, phiplot, pplot, rplot

# if __name__ == "__main__":
short_data = symplot(short)
phugoid_data = symplot(phugoid)

# dutch_data = asymplot(dutch)
# dutch_yawdamp_data = asymplot(dutch_yawdamp)
# aperiodic_data = asymplot(aperiodic)
# spiral_data = asymplot(spiral)
# asymplot(spiral)

from scipy import integrate
def f(x): return x*x
x = np.arange(0.,1.,0.01)

def beta(x, f):
    b = []
    for i in x:
        b.append(integrate.quad(f, 0., i)[0])
    return b