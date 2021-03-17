import analytical_main as am
import numpy as np
import control.matlab as ml
from matplotlib import pyplot as plt


"""
solve equations of motion for inital alpha and velocity
"""

if __name__ == "__main__":
    # ac = am.ac(hp0=5288.28)
    # V = 102.88889
    # t = np.linspace(0,25,250)
    # X0 = np.array([0,0,0,0])
    # u = np.zeros(np.size(t))
    # u[:5] = 0.01
    # sys1 = ac.sym_system(V)
    # y, t, x = ml.lsim(sys1,U=u, T=t,X0=X0)
    #
    # plt.subplot(221)
    # plt.plot(t, np.degrees(y[:,0]))
    # plt.grid()
    #
    # plt.subplot(222)
    # plt.plot(t, np.degrees(y[:,1]))
    # plt.grid()
    #
    # plt.subplot(223)
    # plt.plot(t, np.degrees(y[:,2]))
    # plt.grid()
    #
    # plt.subplot(224)
    # plt.plot(t, np.degrees(y[:,3]))
    # plt.grid()
    #
    # plt.show()

    ac = am.ac(hp0=5288.28)
    V = 102.88889
    sys1 = ac.sym_system(V)
    poles, eigenvectors = np.linalg.eig(ac.A)
    ml.damp(sys1)
    # plt.scatter(np.real(poles), np.imag(poles))
    # plt.grid()
    # plt.show()


