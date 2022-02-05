from neurosim.neurosim import Neuron,Stimulus,Simulation
from matplotlib import pyplot as plt

# Neuron
lif_stdp = Neuron(type='lif',
                    N_exc=2,
                    w_exc=1.0,
                    stdp = True,
                    A_ltp=0.1, tau_ltp=17,
                    A_ltd=-0.05, tau_ltd=34,
                    sra=0.06,
                    ref=1.2)

# Stimuli
rates = [8,5]
stim = Stimulus(stim_type='poisson',
                rate_exc=rates,
                t_sim=15000,
                neuron=lif_stdp)

# Simulation
sim = Simulation(neuron=lif_stdp,stimulus=stim)
sim.simulate()

# Plot potential
sim.plotVoltageTrace()
# plt.savefig('../results/exercise31_04.png')
plt.show()

# Plot firing rate
plt.plot(sim.fr,color='red')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')
# plt.savefig('../results/exercise31_05.png')
plt.show()

# Plot synaptic weights
for r in range(len(rates)):
    c = 'purple'
    if r == 0:
        c = 'green'
    plt.plot(sim.simtime/1000,sim.weights_e[r],label=f'Firing rate : {rates[r]} Hz',color=c)
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.legend()
# plt.savefig('../results/exercise31_06.png')
plt.show()

