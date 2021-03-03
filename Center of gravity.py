import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as sc

meters = 0.0254  # multiply to get meters
kg = 0.453592
g = 9.81
m_dot = 0.048  # kg/s
t = np.arange(0,3000,1)# s
cg_bem = 3  # m

d = pd.read_csv('fuelmass.txt', header=None, delimiter= '&')
mf = np.hstack((d[0],d[2])) * kg  # in kg
mf_x = np.hstack((d[1],d[3]))*100 * kg * meters  # kg*s
x = mf_x/mf  # m

x_mf = sc.interpolate.interp1d(mf,x,kind='cubic')  # m, f(kg)


# --------------------------------------------------------
#                           BEM
# --------------------------------------------------------
BEM = 60500 / g  # [kg]  = to standard aircraft mass?

# --------------------------------------------------------
#                           FUEL
# --------------------------------------------------------
block_fuel = 2700 * kg


# --------------------------------------------------------
#                           PAYLOAD
# --------------------------------------------------------
pilot1 = ["Pilot1", 70, 1]
pilot2 = ["Pilot2", 75, 2]
abattegazzore = ["ABattegazzore", 62, 3]
fbranca = ["FBranca", 58, 4]
yprencipe = ["YPrencipe", 65, 5]
o4 = ["Nout", 80, 6]
o5 = ["Kristen", 80, 7]
o6 = ["ASepulcri", 70, 8]
coordinator = ["Coord", 80, 10]


passengers = pilot1, pilot2, abattegazzore, fbranca, yprencipe, o4, o5, o6, coordinator

seat = []
weight = []
name = []

for i in passengers:
    seat.append(i[2])
    weight.append(i[1])
    name.append(i[0])

loc = []

def getpassengerposition(seat):
    if seat == 1 or seat == 2:
        position = 131 * meters
    if seat == 3 or seat == 4:
        position = 214 * meters
    if seat == 5 or seat == 6:
        position = 251 * meters
    if seat == 7 or seat == 8:
        position = 288 * meters
    if seat == 9 or seat == 10:
        position = 170 * meters
    return position

for i in seat:
    loc.append(getpassengerposition(i))

ramp_mass = BEM + block_fuel + sum(weight)

def mass(time):
    m = BEM + sum(weight) + block_fuel-m_dot*time
    return m

def getfuelx(time):
    mass_fuel = block_fuel-m_dot*time
    x = x_mf(mass_fuel)
    return x

loc = np.array(loc)
weight = np.array(weight)


def cg(time):
    pass_moment = sum(loc*weight)
    fuel_moment = getfuelx(time)*(block_fuel-m_dot*time)
    BEM_moment = BEM * cg_bem
    return (BEM_moment+fuel_moment+pass_moment)/(mass(time))


#plt.plot(t, mass(t))
#plt.plot(mf, x_mf(mf)*mf)
#plt.plot(mf_x, x)
plt.plot(t, cg(t))
plt.show()