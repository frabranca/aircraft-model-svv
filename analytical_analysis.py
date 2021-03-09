import analytical_main2 as am
import numpy as np
import control.matlab as ml
from matplotlib import pyplot as plt

if __name__ == "__main__":
    ac = am.ac(hp0=5288.28)
    V = 102.88889
    t = np.linspace(0,25,250)
    X0 = np.array([0,np.radians(15),0,0])

    sys1 = ac.asym(V)
    y, t = ml.initial(sys1,t,X0)
    print(y[0,:])

    plt.subplot(221)
    #plt.ylim(-2,2)
    plt.plot(t, np.degrees(y[:,0]))
    plt.grid()

    plt.subplot(222)
    #plt.ylim(-20,20)
    plt.plot(t, np.degrees(y[:,1]))
    plt.grid()

    plt.subplot(223)
    #plt.ylim(-2,2)
    plt.plot(t, np.degrees(y[:,2]))
    plt.grid()

    plt.subplot(224)
    #plt.ylim(-2,2)
    plt.plot(t, np.degrees(y[:,3]))
    plt.grid()

    plt.show()

