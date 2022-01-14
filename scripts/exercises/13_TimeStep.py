# --------------------------------------------------------- #
# C11 - COMPUTATIONAL MODELING OF SYNAPTIC PLASTICITY       #
# EXERCISE 1.3 - INTEGRATION TIME STEP                      #
# RUBEN TAMMARO                                             #
# --------------------------------------------------------- #


import matplotlib.pyplot as plt
import numpy as np
from scripts.neurons import g_synapse
from math import exp
# from matplotlib import use
#
# use('Qt5Agg')


# Parameters
from scripts.parameters.param12_LIF_inputs import tau_e,w_ex
t_end = 100
DT = [0.01,0.1,1]

# Analytical solution
time_an = np.linspace(0, t_end, int(t_end / DT[1]))
g_analytical = [exp(-t/tau_e) for t in time_an]

# For plotting multiple zooms
for i in [1,2,3]:

    plt.subplot(1,3,i)
    plt.plot(time_an,g_analytical,label='Analytical')

    # Numerical solution
    for dt in DT:
        time = np.linspace(0, t_end, int(t_end / dt))
        g_num = np.zeros(len(time))
        g_num[0] = 1
        for t in range(len(time)-1):
            g_num[t+1] = g_num[t] + dt*(g_synapse(g_num[t],time[t],[],tau_e,w_ex))

        plt.plot(time,g_num,'--',label='dt = %s'%(dt))


    if i == 1:
        plt.xlim(0,20)
        plt.ylim(0, 1)
    elif i == 2:
        plt.xlim(4,20)
        plt.ylim(0, 0.2)
    else:
        plt.xlim(8,20)
        plt.ylim(0, 0.02)

plt.legend(fontsize='large')
plt.show()