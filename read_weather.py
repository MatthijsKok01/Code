# importing libaries
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math as math
#importing the weather data
data = np.genfromtxt('Almeria_Airp_-hour.dat', skip_header=2,
                     dtype=float,
                     delimiter='\t')

month = data[:,0]
print(month)
hour = data[:,2]
temp_air = data[:,4]
#print(month)
t0 = 0
y0 = [1]
t_end = hour[-1]*3600
p = [1,1]


def heat_bal(t,x,u,p):
    u = u[t]
    A = p[0]
    B = p[1]
    dx = A*x+B*u
    return(dx)





#result = heat_bal(0,2,2,p)
#print(result)
#temp = integrate.RK45(heat_bal(t,y,temp_air,p),t0,y0,t_end)
t0 = 0
y0 = [0]
t_end = hour[-1]
t_span = []
count = -1
for i in hour:
    count +=1
    t_span.append(count * 3600)
p = [1,1]
def make_input(t,y):
    time_index = t/3600 + 0*y
    time_index = int(time_index)
    u = month[time_index]
    print(u)
    return u
def heat_bal2(t,y,p):
    #creating the measured values at time t as input
    time_index = t/3600 + 0*y
    time_index = int(time_index)
    u = month[time_index]
    dxdt = np.array(u) -1#math.exp(-y)
    #print(dxdt)
    return dxdt

solver = integrate.solve_ivp(heat_bal2,[0, t_span[-1]] ,y0,t_eval=t_span, args = [0.0000001])
print(solver)

t_vals = solver.t
y_vals = solver.y
y_vals = y_vals[0]


##          plot results
plt.plot(t_vals, y_vals)
plt.show()