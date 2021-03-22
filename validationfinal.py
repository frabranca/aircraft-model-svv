import numpy as np
from flightdataprocessing import approx, short_data, dutch_yawdamp_data
import Numerical_main as nm
from matplotlib import pyplot as plt
import seaborn as sns

class modesym():
    def __init__(self, data=short_data):
        self.short_label = ["$\hat{u}$", r"$\alpha$", r"$\theta$", "qc/V"]

        # state variables
        self.cond, self.tplot, self.uplot, self.aplot, self.thplot, self.qplot, self.deplot = data
        self.t = np.linspace(0, (self.tplot[-1] - self.tplot[0]),np.size(self.tplot))

        # initial conditions
        self.m0 = self.cond[0][0]
        self.h0 = self.cond[1][0]
        self.V0 = self.cond[2][0]
        self.t0 = self.cond[3][0]
        self.u0 = self.cond[4][0]
        self.a0 = self.cond[5][0]
        self.th0 = self.cond[6][0]
        self.q0 = self.cond[7][0]
        self.x0 = np.array([self.u0, self.a0, self.th0, self.q0])
        # self.totalplot = np.zeros((4,np.size(self.tplot)))
        # self.totalplot[0,:] = self.uplot
        # self.totalplot[1,:] = self.aplot
        # self.totalplot[2,:] = self.thplot
        # self.totalplot[3,:] = self.qplot

    def plot(self):
        # NUMERICAL MODEL STATE VARIABLES
        ac = nm.ac(m = self.m0, initial = self.x0, hp0 = self.h0, V0 = self.V0)
        self.y = ac.sym_input_response(self.t, self.deplot, self.x0)

        # plot everything
        sns.set_theme()
        plt.figure(figsize=(12,8))
        f = 20

        plt.subplot(221)
        # plt.title("de")
        # plt.plot(short.t, short.deplot)
        plt.xlabel('Time [min]')
        plt.ylabel(self.short_label[0] + str(' [-]'))
        plt.plot(self.t, self.y[0].T[0], label='numerical model')
        plt.plot(self.t, self.uplot, label='flight data')

        plt.subplot(222)
        plt.xlabel('Time [min]')
        plt.ylabel(self.short_label[1] + str(' [rad]'))
        plt.plot(self.t, self.y[0].T[1], label='numerical model')
        plt.plot(self.t, self.aplot, label='flight data')

        plt.subplot(223)
        plt.xlabel('Time [min]')
        plt.ylabel(self.short_label[2] + str(' [rad]'))
        plt.plot(self.t, self.y[0].T[2], label='numerical model')
        plt.plot(self.t, self.thplot, label='flight data')

        plt.subplot(224)
        plt.xlabel('Time [min]')
        plt.ylabel(self.short_label[3] + str(' [rad/s]'))
        plt.plot(self.t, self.y[0].T[3], label='numerical model')
        plt.plot(self.t, self.qplot, label='flight data')
        plt.legend(loc='best')

        plt.figure(figsize=(12,8))
        plt.xlabel('Time [s]', fontsize=f)
        plt.ylabel('Elevator deflection [rad]', fontsize=f)
        plt.plot(self.t, self.deplot)
        plt.show()

modesym = modesym()
modesym.plot()
# error = sum( abs( short.uplot - y[0].T[0] ) )
# print(ac.CLa)
# print(error)

class modeasym():
    def __init__(self, data=dutch_yawdamp_data):
        self.dutch_yawdamp_label = [r"$\beta$", r"$\phi$", "pb/2V", "rb/2V"]

        # state variables
        self.cond, self.tplot, self.phiplot, self.rplot, self.pplot, self.daplot, self.drplot = data
        self.t = np.linspace(0, (self.tplot[-1] - self.tplot[0]),np.size(self.tplot))

        # initial conditions
        self.m0 = self.cond[0][0]
        self.h0 = self.cond[1][0]
        self.V0 = self.cond[2][0]
        self.t0 = self.cond[3][0]
        self.beta0 = self.cond[4]
        self.phi0 = self.cond[5][0]
        self.p0 = self.cond[6][0]
        self.r0 = self.cond[7][0]
        self.x0 = np.array([self.beta0, self.phi0, self.p0, self.r0])
        # self.totalplot = np.zeros((4,np.size(self.tplot)))
        # self.totalplot[0,:] = self.uplot
        # self.totalplot[1,:] = self.aplot
        # self.totalplot[2,:] = self.thplot
        # self.totalplot[3,:] = self.qplot

    def plot(self):
        # NUMERICAL MODEL STATE VARIABLES
        ac = nm.ac(m = self.m0, initial = self.x0, hp0 = self.h0, V0 = self.V0)
        self.input = np.zeros((np.size(self.t),2))
        self.input[:,0] = self.daplot
        self.input[:,1] = self.drplot
        self.y = ac.asym_input_response(self.t, -self.input, self.x0)

        # plot everything
        sns.set_theme()
        plt.figure(figsize=(12,8))
        f = 20

        plt.subplot(221)
        # plt.title("de")
        # plt.plot(short.t, short.deplot)
        # plt.xlabel('Time [min]')
        # plt.ylabel(self.dutch_yawdamp_label[0] + str('[-]'))
        # plt.plot(self.t, y[0].T[0], label='numerical model')
        # plt.plot(self.t, self., label='flight data')

        plt.subplot(222)
        plt.xlabel('Time [min]')
        plt.ylabel(self.dutch_yawdamp_label[1] + str(' [rad]'))
        plt.plot(self.t, self.y[0].T[1], label='numerical model')
        plt.plot(self.t, self.phiplot, label='flight data')

        plt.subplot(223)
        plt.xlabel('Time [min]')
        plt.ylabel(self.dutch_yawdamp_label[2] + str(' [rad/s]'))
        plt.plot(self.t, self.y[0].T[2], label='numerical model')
        plt.plot(self.t, self.pplot, label='flight data')

        plt.subplot(224)
        plt.xlabel('Time [min]')
        plt.ylabel(self.dutch_yawdamp_label[3] + str(' [rad/s]'))
        plt.plot(self.t, self.y[0].T[3], label='numerical model')
        plt.plot(self.t, self.rplot, label='flight data')
        plt.legend(loc='best')

        plt.figure(figsize=(12,8))
        plt.xlabel('Time [s]', fontsize=f)
        plt.ylabel('Rudder [rad]', fontsize=f)
        plt.plot(self.t, self.drplot)
        plt.show()
modeasym = modeasym()
modeasym.plot()