import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as sc
import seaborn as sns

sns.set_theme()

fd = pd.read_excel('fuel-only-data.xlsx')
t = fd['t'].values
fu = fd['fu1'].values + fd['fu2'].values


meters = 0.0254  # multiply to get meters
kg = 0.453592
g = 9.81
m_dot = 0.048  # kg/s
#t = np.arange(0, 86*60, 1)# s
cg_bem = 291.65 * meters # m
req = [276.1*meters, 285.8*meters]

fu = fu * kg

t_bchange = (51*60 + 44)
begin_ind = np.where(t == t_bchange)[0][0]

t_echange = (53*60 + 40)
end_ind = np.where(t == t_echange)[0][0]

mac = 80.98 * meters
x_ac = 261.56 * meters + mac/4

d = pd.read_csv('fuelmass.txt', header=None, delimiter= '&')
mf = np.hstack((d[0],d[2])) * kg  # in kg
mf_x = np.hstack((d[1],d[3]))*100 * kg * meters  # kg*s
x = mf_x/mf  # m

x_mf = sc.interpolate.interp1d(mf, x, kind='cubic')  # m, f(kg)

# --------------------------------------------------------
#                           BEM
# --------------------------------------------------------
BEM = 9165.0*kg  # [kg]  = to standard aircraft mass?

# --------------------------------------------------------
#                           FUEL
# --------------------------------------------------------
block_fuel = 2700 * kg

def fuelmass(time):
    ind = np.where(t == time)
    mass_fuel = block_fuel - fu[ind]
    return mass_fuel


def getfuelx(time):
    x_fuel = x_mf(fuelmass(time))
    return x_fuel


# --------------------------------------------------------
#                           PAYLOAD
# --------------------------------------------------------
def passfunc(kloc):
    pilot1 = ["Pilot1", 82, 1]
    pilot2 = ["Pilot2", 98, 2]
    o1 = ["YPrencipe", 59, 3]
    o2 = ["FBranca", 60, 4]
    o3 = ["ABattegazzore", 63, 5]
    o4 = ["Nout", 75, 6]
    o5 = ["ASepulcri", 86, 7]
    o6 = ["Kirsten", 89, kloc]
    coordinator = ["Coord", 79, 10]


    passengers = pilot1, pilot2, o1, o2, o3, o4, o5, o6, coordinator

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

    return weight, loc


ramp_mass = BEM + block_fuel + sum(passfunc(8)[0])


def mass(time):
    m = BEM + sum(passfunc(8)[0]) + fuelmass(time)
    return m


def cg_normal(time):
    weight = np.array(passfunc(8)[0])
    loc = np.array(passfunc(8)[1])
    pass_moment = sum(loc*weight)
    fuel_moment = getfuelx(time)*(fuelmass(time))
    BEM_moment = BEM * cg_bem
    return (BEM_moment+fuel_moment+pass_moment)/(mass(time))


cg_normal_list = cg_normal(t)


def cg_change(time):
    weight = np.array(passfunc(9)[0])
    loc = np.array(passfunc(9)[1])
    pass_moment = sum(loc * weight)
    fuel_moment = getfuelx(time) * (fuelmass(time))
    BEM_moment = BEM * cg_bem
    return (BEM_moment + fuel_moment + pass_moment) / (mass(time))


cg_change_list = cg_change(t)
cg_final = np.copy(cg_normal_list)
cg_final[begin_ind:end_ind] = cg_change_list[begin_ind:end_ind]

plt.figure(1)
plt.subplot(121)
plt.title("CG excursion over time")
plt.xlabel("Time [s]")
plt.ylabel("CG_x [m]")
#plt.plot(t, cg_change_list, label="8 to 9")
plt.plot(t, cg_final, color="green")


plt.subplot(122)
plt.title("Aircraft's mass over time")
plt.xlabel("Time [s]")
plt.ylabel("Mass [kg]")
plt.plot(t, mass(t), color="red")

plt.show()

print(req)
print("CG x coordinates at selected points of the flight")
print("-------------------------------------------------")
print("start: ", cg_final[0], "end: ", cg_final[-1])
print("right before kirsten moves: ", cg_normal_list[begin_ind], "right after kirsten moves: ", cg_change_list[begin_ind])
print("right before kirsten moves back: ", cg_change_list[end_ind], "right after kirsten moves back: ", cg_normal_list[end_ind])

