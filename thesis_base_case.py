# importing libaries
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import math as math
#importing the weather data
data = pd.read_csv('Almeria_Airp_-hour.dat',delimiter='\t')
data = np.genfromtxt('Almeria_Airp_-hour.dat', skip_header=2,
                     dtype=float,
                     delimiter='\t')

print((data[0]))
data = data.T
[month, dofm, dofy, hi, Ta, G_Gh, Td, RH, G_DH, FF, DD, Lin, RR, Sd, N, hs, TL, G_Bn, G_Gc, G_DC, GEX, G_Ghmod, PAR, Snd, Lup] = data
to_transpose = [month, dofm, dofy, hi, Ta, G_Gh, Td, RH, G_DH, FF, DD, Lin, RR, Sd, N, hs, TL, G_Bn, G_Gc, G_DC, GEX, G_Ghmod, PAR, Snd, Lup]
'''dofm: day of month
    dofy: day of the year
    hi: hour of the day
    Ta: temperature of air
    G_Gh: average global horizontal radiation?
    Td: dew point temperature
    RH: relative himidity (%)
    G_DH: diffuse radiation (horizontal
    FF: wind speed (m/s)
    DD: wind direction
    Lin: longwave horizontal radiation impinging 
    RR: preciptation
    Sd: effective sunshine duration
    N: cloud cover
    hs: solar altitude
    TL: linke turbidity factor
    G_Bn: direct normal radiation
    G_Gc: global radiation horizontal maximum. 
    GEX: Extraterrestrial radiation
    G_Ghmod: 
    PAR: Photosynthetically active radiation
    Snd: 
    Lup: Longwave (thermal, infrared) radiation on horizontal surface emitted from the earth's surface (longwave 
        outgoing) wavelength > 3 µm
'''
print(month.shape)
time_scale = 3600
hour = hi
temp_air = Ta
#print(month)
t0 = 0
y0 = [1]
t_end = hour[-1]*time_scale
p = [1, 1]



#temp = integrate.RK45(heat_bal(t,y,temp_air,p),t0,y0,t_end)

t_span = []
count = -1
for i in hour:
    count +=1
    t_span.append(count * time_scale)
p = [1,1]


def make_input(t,y):
    time_index = t/time_scale + 0*y
    time_index = int(time_index)
    u = month[time_index]
    print(u)
    return u


def heat_bal(t,y,p):
    '''creating the measured values at time t as input'''
    time_index = t/3600 + 0*y
    time_index = int(time_index)
    u = month[time_index]
    dxdt = np.array(u) - 1
    return dxdt


solver = integrate.solve_ivp (heat_bal,[0, t_span[-1]], y0, t_eval=t_span, args = [0.0000001])
print(solver)

t_vals = solver.t
y_vals = solver.y
y_vals = y_vals[0]


##          plot results
plt.plot(t_vals, y_vals)
plt.show()