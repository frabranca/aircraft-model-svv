import numpy as np

inch = 0.0254  # multiply to get meters
pound = 0.453592
g = 9.81
m_dot = 0.048  # kg/s
t = 0  # s

# --------------------------------------------------------
#                           BEM
# --------------------------------------------------------
BEM = 60500 / g  # [kg]  = to standard aircraft mass?

# --------------------------------------------------------
#                           FUEL
# --------------------------------------------------------
block_fuel = 2700 * pound # kg max fuel flow?

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
#passengers = np.matrix(passengers)

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
        position = 131 * inch
    if seat == 3 or seat == 4:
        position = 214 * inch
    if seat == 5 or seat == 6:
        position = 251 * inch
    if seat == 7 or seat == 8:
        position = 288 * inch
    if seat == 9 or seat == 10:
        position = 170 * inch
    return position

for i in seat:
    loc.append(getpassengerposition(i))

ramp_mass = BEM + block_fuel + sum(weight)

mass = ramp_mass - m_dot * t

cg_x =