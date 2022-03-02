from neurosim.neurosim import Neuron,Stimulus,Simulation
from matplotlib import pyplot as plt
from matplotlib import rcParams, use
import numpy as np
from time import time as tic

#%%

# Parameters
rcParams['figure.figsize'] = (10,4)
use('tkagg')
t_sim = 4000000
dt = 0.1

# Neuron
st = tic()
lif = Neuron(type='lif',
             N_exc=10, w_exc=0.35,
             N_inh=10, w_inh=1.0,
             stdp='e', A_ltp_e=0.001, A_ltd_e=-0.0005,
             ip = 'e', r_target= 3, eta_ip =  0.1
             )

# Stimulus 1 excitatory 0.1
stim1 = Stimulus(stim_type='poisson',
                t_sim=t_sim,
                dt = dt,
                S_e=5, rate_exc=3,
                corr_type='exp',
                corr=0.1,
                neuron=lif)

# Stimulus 2 excitatory 0.2
stim2 = Stimulus(stim_type='poisson',
                t_sim=t_sim,
                 dt = dt,
                S_e=5, rate_exc=3,
                corr_type='exp',
                corr=0.2,
                neuron=lif)

# Stimulus 3 inhibitory
stim3 = Stimulus(stim_type='poisson',
                t_sim=t_sim,
                 dt= dt,
                S_i=10, rate_exc=10,
                neuron=lif)

#%%

# Simulation
stim = [stim1,stim2,stim3]
sim = Simulation(neuron=lif,stimulus=stim)
sim.simulate()

toc = tic() - st

print('\n\n',toc)


# %% PLOTS

# SYNAPTIC WEIGHTS
plt.subplot(1,3,1)
# Group 1
for w in range(sim.stim_group_boundaries):
    plt.plot(sim.simtime/1000,sim.weights_e[w],color='tab:green',label='Group 1 : c = 0.1')

# Group 2
for w in range(sim.stim_group_boundaries, len(sim.weights_e)):
    plt.plot(sim.simtime/1000,sim.weights_e[w],color='tab:purple',label='Group 2 : c = 0.2')
plt.xlabel('Time (s)')
plt.ylabel('Synaptic weights ')
plt.title('Synaptic weights', fontweight='bold')


# THRESHOLD
plt.subplot(1,3,2)
plt.plot(sim.simtime/1000,sim.theta)
plt.xlabel('Time (s)')
plt.ylabel('Potential (mV)')
plt.title('Threshold potential', fontweight='bold')


# FIRING RATE
plt.subplot(1,3,3)
sim.plotFiringRate()

plt.show()



# %%

# use('tkagg')
rcParams['figure.figsize'] = (12,8)

# xlims = [4000, 2000, 1000, 500, 250, 100, 50]
xlims = [100, 250, 500, 1000, 4000]

for i, lim in enumerate(xlims):

    plt.subplot(2,len(xlims),i+1)
    plt.plot(sim.simtime/1000,sim.theta)
    plt.xlim(0,lim)
    if i == 0:
        plt.ylabel('Spiking threshold potential (mV)')


    plt.subplot(2,len(xlims),len(xlims)+i+1)
    plt.plot(sim.simtime / 1000, sim.fr, color='red', lw=2)
    plt.xlim(0, lim)
    if i == 0:
        plt.ylabel('Firing rate (Hz)')
    if i == 2:
        plt.xlabel('Time (s)')

plt.suptitle('Intrinsic Plasticity',fontweight='bold')
plt.show()