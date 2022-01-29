from neurosim.neurosim import Neuron
from neurosim.neurosim import Stimulus
from neurosim.neurosim import Simulation
from matplotlib import pyplot as plt



# ------------- FIGURE 1 ------------- #
# Excitatory synaptic input

# Initialize neuron with 1 excitatory synapse
lif = Neuron(type='LIF', N_exc=1, E_exc=0, tau_e=3, w_exc=0.5)

# Generate periodic stimulus at 6 Hz
stim = Stimulus(stim_type='periodic', rate_exc=[6],
                t_sim=1000, neuron=lif)

# Create simulation
sim = Simulation(neuron=lif,stimulus=stim,t_sim=1000)

# Simulate
sim.simulate()

# Plot
plt.plot(sim.simtime,sim.potential,label='We = %s'%lif.w_exc)
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
# plt.savefig('../results/exercise12_01.png')
plt.show()


# ------------- FIGURE 2 ------------- #
# Different synaptic weights

W = [0.5, 1.5, 3.0]

for i,w in enumerate(W):

    # Initialize neuron with 1 exc. synapse
    lif = Neuron(type='LIF', N_exc=1, E_exc=0, tau_e=3, w_exc=w)

    # Generate periodic stimulus at 6 Hz
    stim = Stimulus(stim_type='periodic', rate_exc=[6],
                    t_sim=1000, neuron=lif)
    # Create simulation
    sim = Simulation(neuron=lif,stimulus=stim,t_sim=1000)

    # Simulate
    sim.simulate()

    # Plot
    plt.subplot(len(W),1,i+1)
    plt.plot(sim.simtime,sim.potential,label='We = %s'%lif.w_exc)
    plt.legend(loc='upper left',fontsize='x-small')
    if i + 1 == len(W):
        plt.xlabel('Time (ms)')
    if i + 1 == round(len(W)/2):
        plt.ylabel('Membrane potential (mV)')

# plt.savefig('../results/exercise12_02.png')
plt.show()


# ------------- FIGURE 3 ------------- #
# Inhibitory synaptic input added

# Initialize neuron with 1 excitatory and 1 inhibitory synapses
lif = Neuron(type='LIF',
             N_exc=1, w_exc=3.0,
             N_inh=1, w_inh=3.0)

# Generate periodic stimulus at 6 Hz and 3 Hz
stim = Stimulus(stim_type='periodic',
                rate_exc=6, rate_inh=3,
                t_sim=1000, neuron=lif)

# Create simulation
sim = Simulation(neuron=lif,stimulus=stim)

# Simulate
sim.simulate()

# Plot
plt.subplot(2,1,1)
plt.plot(sim.simtime,sim.potential,label='We = %s'%lif.w_exc)
plt.legend()

plt.ylabel('Membrane potential (mV)')

plt.subplot(2,1,2)
plt.plot(sim.simtime,sim.g_exc[0],color='tab:green')
plt.plot(sim.simtime,-sim.g_inh[0],color='tab:purple')
plt.ylabel('Conductance (mS)')
plt.xlabel('Time (ms)')
# plt.savefig('../results/exercise12_03.png')
plt.show()



