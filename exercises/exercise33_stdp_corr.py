from neurosim.neurosim import Neuron,Stimulus,Simulation
from matplotlib import pyplot as plt
from numpy import mean

# -------------------- PARAMETERS -------------------- #
# Change to obtain :        # Figure 1      # Figure 2      # Figure 3      # Figure 4      # Figure 5      # Figure 6      #       # Figure 7          #       # Figure 8          #       # Figure 9          #       # Figure 10         #       # Figure 11         #       # Figure 12         #       # Figure 13         #
c1   = 0.1                  #   0.1         |   0.1         |   0.1         |   0.1         |   0.1         |   0.1         |           0.1             |           0.1             |           0.1             #           0.1             |           0.1             |           0.1             |           0.1             #
c2   = 0.1                  #   0.1         |   0.2         |   0.5         |   0.1         |   0.2         |   0.5         |           0.1             |           0.2             |           0.5             #           0.1             |           0.2             |           0.5             |           0.9             #
c    = 'inst'               #  'inst'       |  'inst'       |  'inst'       |  'exp'        |  'exp'        |  'exp'        |          'inst'           |          'inst'           |          'inst'           #          'exp'            |          'exp'            |          'exp'            |          'exp'            #
we   = 0.5                  #   0.5         |   0.5         |   0.5         |   0.5         |   0.5         |   0.5         |   [1.0]*10 + [0.5]*10     |   '[1.0]*10 + [0.5]*10'   |   '[1.0]*10 + [0.5]*10'   #   [1.0]*10 + [0.5]*10     |   '[1.0]*10 + [0.5]*10'   |   '[1.0]*10 + [0.5]*10'   |   '[1.0]*10 + [0.5]*10'   #
fign = '01'
t_sim = 30000

# -------------------- NEURONS -------------------- #

lif = Neuron(type='lif',
             N_exc=20,w_exc=we,
             N_inh=10,w_inh=1.0,
             stdp='e', A_ltp_e=0.02, A_ltd_e=-0.01)

# -------------------- STIMULI -------------------- #

# GROUP1 STIMULUS
group1 = Stimulus(stim_type='poisson',
                S_e = 10,
                S_i = 10,
                rate_exc=10,
                rate_inh=10,
                corr_type=c,
                corr=c1,
                t_sim=t_sim,
                neuron=lif)

# GROUP2 STIMULUS
group2 = Stimulus(stim_type='poisson',
                S_e = 10,
                rate_exc=10,
                corr_type=c,
                corr=c2,
                t_sim=t_sim,
                neuron=lif)

stim = [group1, group2]

# -------------------- SIMULATIONS -------------------- #

sim = Simulation(neuron=lif, stimulus=stim)
sim.simulate()


# -------------------- PLOTS -------------------- #

# Title
if c == 'exp':
    tit = 'Exponential correlation'
else:
    tit = 'Instantaneous correlation'

# Plot firing rate
sim.plotFiringRate(show=False)
plt.title(f'Firing rate - {tit}',fontweight='bold')
# plt.savefig(f'../results/exercise33_FR_{fign}.png')
plt.show()


# Plot group 1
for w in range(sim.stim_group_boundaries):
    plt.plot(sim.simtime/1000,sim.weights_e[w],color='tab:green')

# Plot group 1
for w in range(sim.stim_group_boundaries, len(sim.weights_e)):
    plt.plot(sim.simtime/1000,sim.weights_e[w],color='tab:purple')

# Plot averages
sim.plotSynapticWeights(syn='i',avg=True)
plt.plot(sim.simtime / 1000, mean(sim.weights_e[:sim.stim_group_boundaries - 1], axis=0), color="green", lw=5, label=f'Excitatory (c : {c1})')
plt.plot(sim.simtime / 1000, mean(sim.weights_e[sim.stim_group_boundaries:], axis=0), color="purple", lw=5, label=f'Excitatory (c : {c2})')

# Labels
plt.title(f'Weight evolution - {tit}',fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('Synaptic strength')
plt.legend()
# plt.savefig(f'../results/exercise33_W_{fign}.png')
plt.show()