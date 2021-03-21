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

    xx_avg = np.average(xx, axis=1)

    xx = (np.ones((np.shape(xx))).T * xx_avg).T
    xx = np.concatenate(xx)
    tt = np.concatenate(tt)

    return tt, xx

def symplot(mode):
    start = mode[0]
    end = mode[1]

    # time
    tplot = t[(t >= start) & (t <= end)] #s
    # velocity
    vt = tas[(t >= start) & (t <= end)] #m/s

    # initial conditions
    c = 2.0569
    ind = np.where(t == tplot[0])[0]
    m0 = m[ind] # kg
    h0 = hp[ind] # m
    V0 = tas[ind] #m/s
    t0 = t[ind] #s
    u0 = (vt[0]-V0)/V0 # -
    a0 = a[ind] #rad
    th0 = theta[ind] #rad
    q0 = q[ind]*c/V0 #rad

    # state variables
    uplot = (vt - V0)/V0 # -
    aplot = a[(t >= start) & (t <= end)]
    thplot = theta[(t >= start) & (t <= end)]
    qplot = q[(t >= start) & (t <= end)] * c/V0
    deplot = de[(t >= start) & (t <= end)]

    x0 = np.array([m0, h0, V0, t0, u0, a0, th0, q0])
    return x0, tplot, uplot, aplot, thplot, qplot, deplot


def asymplot(mode):
    sns.set_theme()
    start = mode[0]
    end = mode[1]

    # time
    tplot = t[(t >= start) & (t <= end)] # s
    mplot = m[(t >= start) & (t <= end)] # kg

    # initial conditions
    b = 15.911
    ind = np.where(t == tplot[0])[0]
    m0 = mplot[ind] # kg
    h0 = hp[ind] # m
    V0 = tas[ind] # m/s
    t0 = t[ind] # s
    beta0 = 0.
    phi0 = phi[ind]
    p0 = p[ind]*b/V0/2
    r0 = r[ind]*b/V0/2

    tplot = tplot

    phiplot = phi[(t >= start) & (t <= end)]
    pplot = p[(t >= start) & (t <= end)] *b/V0/2
    rplot = r[(t >= start) & (t <= end)] *b/V0/2
    daplot = da[(t >= start) & (t <= end)]
    drplot = dr[(t >= start) & (t <= end)]

    x0 = np.array([m0, h0, V0, t0, beta0, phi0, p0, r0])
    return x0, tplot, phiplot, pplot, rplot

# short_data = symplot(short)
phugoid_data = symplot(phugoid)

# dutch_data = asymplot(dutch)
# dutch_yawdamp_data = asymplot(dutch_yawdamp)
# aperiodic_data = asymplot(aperiodic)
# spiral_data = asymplot(spiral)
# asymplot(spiral)

if __name__ == "__main__":
    x = np.linspace(0,10,1000)
    y = np.sin(x)
    xav, yav = approx(x,y)
    print(np.size(y), np.size(yav))
    plt.plot(xav,yav)
    plt.show()
    from scipy import integrate
    def f(x): return x*x
    x = np.arange(0.,1.,0.01)

    def beta(x, f):
        b = []
        for i in x:
            b.append(integrate.quad(f, 0., i)[0])
        return b