from neurosim.neurosim import Neuron,Stimulus
from matplotlib import pyplot as plt
import numpy as np


lif = Neuron(type='LIF',N_exc=1,w_exc=3.0)

stim = Stimulus(type='periodic',rate_exc=0,neuron=lif)

# Analytical solution
t_end = 100
DT = [0.01,0.1,1]
time_analytical = np.linspace(0, t_end, int(t_end / DT[1]))
g_analytical = [np.exp(-t/lif.tau_e) for t in time_analytical]




# For plotting multiple zooms
for i in [1,2,3]:

    plt.subplot(1,3,i)
    plt.plot(time_analytical, g_analytical, label='Analytical', linewidth=2, color='purple')


    # Numerical solution
    for dt in DT:
        time = np.linspace(0, t_end, int(t_end / dt))
        g_num = np.zeros(len(time))
        g_num[0] = 1
        for t in range(len(time)-1):
            g_num[t+1] = g_num[t] + dt*(lif.synapse('e',g_num[t],t,stim,0))

        plt.plot(time,g_num,'--',label='dt = %s'%(dt))



    if i == 1:
        plt.xlim(0,20)
        plt.ylim(0, 1)
    elif i == 2:
        plt.xlim(4,20)
        plt.ylim(0, 0.2)
        plt.legend(fontsize='medium', loc="upper left")
    else:
        plt.xlim(11,15)
        plt.ylim(0.008, 0.0125)

# plt.savefig('../results/exercise13.png')
plt.show()
