import pandas as pd
import numpy as np
import scipy.integrate as int

fd = pd.read_csv('data.csv')
t = fd['t'].values

# SYMMETRIC STATE VARIABLES
tas = fd['tas'].values * 0.514444       # True airspeed
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
short = np.array([3657+2.3, 3670])
phugoid = np.array([3344, 3500])
dutch = np.array([3542, 3572])
dutch_yawdamp = np.array([3595, 3615])
aperiodic = np.array([3759, 3789])
spiral = np.array([3840, 3840+100])
search  = np.array([3840, 3840+100])

def approx(t, x, step=35):
    xx = np.array(x)
    tt = np.array(t)

    n = np.size(tt)%step
    xx = xx[0:-n]
    tt = tt[0:-n]

    xx = xx.reshape(int(np.size(xx)/step), step)
    tt = tt.reshape(int(np.size(tt)/step), step)

    xx_avg = np.average(xx, axis=1)

    xx = (np.ones((np.shape(xx))).T * xx_avg).T
    xx = np.concatenate(xx)
    tt = np.concatenate(tt)

    return tt, xx

def symplot(mode):
    start = mode[0]
    end = mode[1]

    # time
    tplot = t[(t >= start) & (t <= end)] # s
    # velocity
    vt = tas[(t >= start) & (t <= end)] # m/s

    # initial conditions
    c = 2.0569
    ind = np.where(t == tplot[0])[0]
    m0 = m[ind] # kg
    h0 = hp[ind] # m
    V0 = tas[ind] #m/s
    t0 = t[ind] #s
    u0 = (vt[0]-V0)/V0 # -
    a0 = np.array([0.0]) #rad
    th0 = np.array([0.0]) #rad
    q0 = q[ind]*c/V0 #rad

    # state variables
    uplot = (vt - V0)/V0 # -
    aplot = a[(t >= start) & (t <= end)] - a[ind]
    thplot = theta[(t >= start) & (t <= end)] - theta[ind]
    qplot = q[(t >= start) & (t <= end)] * c/V0

    # input
    deplot = de[(t >= start) & (t <= end)]

    x0 = np.array([m0, h0, V0, t0, u0, a0, th0, q0])
    return x0, tplot, uplot, aplot, thplot, qplot, deplot


def asymplot(mode):
    start = mode[0]
    end = mode[1]

    # time
    tplot = t[(t >= start) & (t <= end)] #s

    # initial conditions
    b = 15.911
    ind = np.where(t == tplot[0])[0]
    m0 = m[ind] # kg
    h0 = hp[ind] # m
    V0 = tas[ind] # m/s
    t0 = t[ind] # s
    beta0 = 0.  # rad
    phi0 = np.array([0.0]) # rad
    r0 = np.array([0.0]) # r[ind]*b/V0/2 # #rad
    p0 = np.array([0.0]) # p[ind]*b/V0/2 #rad

    # state variables
    phiplot = phi[(t >= start) & (t <= end)] - phi[ind]
    rplot = (r[(t >= start) & (t <= end)] - r[ind])* b/(2*V0)
    pplot = (p[(t >= start) & (t <= end)] - p[ind])* b/(2*V0)

    # inputs
    daplot = -da[(t >= start) & (t <= end)]  + da[ind]
    drplot = -dr[(t >= start) & (t <= end)]  + dr[ind]

    x0 = np.array('dtype=object', [m0, h0, V0, t0, beta0, phi0, p0, r0])
    return x0, tplot, phiplot, rplot, pplot, daplot, drplot

short_data = symplot(short)
phugoid_data = symplot(phugoid)
dutch_data = asymplot(dutch)
dutch_yawdamp_data = asymplot(dutch_yawdamp)
aperiodic_data = asymplot(aperiodic)
search_data = asymplot(search)
dutch_data = asymplot(dutch)
dutch_yawdamp_data = asymplot(dutch_yawdamp)
aperiodic_data = asymplot(aperiodic)
spiral_data = asymplot(spiral)
