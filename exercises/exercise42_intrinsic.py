from neurosim.neurosim import Neuron,Stimulus,Simulation
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np

#%%
rcParams['figure.figsize'] = (10,4)

t_sim = 100000

lif = Neuron(type='lif',
             N_exc=10, w_exc=0.35,
             N_inh=10, w_inh=1.0,
             stdp='e', A_ltp_e=0.001, A_ltd_e=-0.0005,
             ip = 'e', r_target= 3, eta_ip =  0.1
             )

stim1 = Stimulus(stim_type='poisson',
                t_sim=t_sim,
                S_e=5, rate_exc=3,
                corr_type='inst',
                corr=0.1,
                neuron=lif)

stim2 = Stimulus(stim_type='poisson',
                t_sim=t_sim,
                S_e=5, rate_exc=3,
                corr_type='inst',
                corr=0.2,
                neuron=lif)

stim3 = Stimulus(stim_type='poisson',
                t_sim=t_sim,
                S_i=10, rate_exc=10,
                neuron=lif)



stim = [stim1,stim2,stim3]
sim = Simulation(neuron=lif,stimulus=stim)
sim.simulate()

# %%

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





plt.subplot(1,3,2)
plt.plot(sim.simtime/1000,sim.theta)
plt.xlabel('Time (s)')
plt.ylabel('Potential (mV)')
plt.title('Threshold potential', fontweight='bold')


plt.subplot(1,3,3)
sim.plotFiringRate()

plt.show()


sim.plotVoltageTrace(show=True)