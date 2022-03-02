from neurosim.neurosim import Neuron,Stimulus,Simulation
from matplotlib import pyplot as plt, rcParams, use
import numpy as np

# use('tkagg')
# ----------------  PARAMETERS ---------------- #

# Parameters        # FIGURE 1      # FIGURE 2      # FIGURE 3      # FIGURE 4
c1 = 0.1            #               #               #               #
c2 = 0.2            #               #               #               #
t_sim = 100000      #               #               #               #
Ne = 100            #               #               #               #
Ni = 30             #               #               #               #
rate = 10           #               #               #               #
corr = 'inst'       # 'inst'        #   'exp'       #   'inst'      #   'exp'
wtot = 3            #   3           #     3         #     13        #     13
fn = '10'           #   '01         #    '02'       #    '03'       #    '04'

rcParams['figure.figsize'] = (16,12)

titl = 'Instantaneous Correlation'
if corr == 'exp':
    titl = 'Exponential Correlation'
else:
    titl = 'Instantaneous Correlation'

# ----------------  SIMULATION ---------------- #

# Neuron
lif = Neuron(type='lif',
             N_exc=Ne, w_exc=0.1,
             N_inh=Ni, w_inh=1.0,
             stdp='e',A_ltp_e=0.02, A_ltd_e=-0.01,
             normalize='e', # Synaptic normalization on excitatory synapses only
             eta_norm=0.2,       # Normalization rate
             W_tot = wtot,     # Maximum allowed weight
             t_norm=1000)   # Normalization event time step (ms)


# Excitatory synapses - Group 1
stim1 = Stimulus(stim_type='poisson',
                 S_e = int(Ne/2),
                 S_i = 0,
                 rate_exc=rate,
                 corr_type=corr,
                 corr=c1,
                 t_sim=t_sim,
                 neuron=lif)

# Excitatory synapses - Group 2
stim2 = Stimulus(stim_type='poisson',
                 S_e = int(Ne/2),
                 S_i = 0,
                 rate_exc=rate,
                 corr_type=corr,
                 corr=c2,
                 t_sim=t_sim,
                 neuron=lif)

# Inhibitory synapses
stim3 = Stimulus(stim_type='poisson',
                 S_e = 0,
                 S_i = Ni,
                 rate_exc=10,
                 t_sim=t_sim,
                 neuron=lif)

# Simulation
stimuli = [stim1,stim2,stim3]
sim = Simulation(neuron=lif,stimulus=stimuli)
sim.simulate()

# ----------------  PLOTS ---------------- #

rcParams['figure.figsize'] = (16,14)
# ----------------  PLOTS ---------------- #


plt.subplot2grid((4,3),(0,0),colspan=2)

# PLOT FIRING RATE
plt.plot(np.linspace(1,len(sim.fr[1:]),len(sim.fr[1:])),sim.fr[1:],color='red',lw=2)
plt.ylabel('Firing rate (Hz)')
plt.title(f'Firing rate',fontweight='bold')

# Plot WTOT
plt.subplot2grid((4,3),(1,0),colspan=2)
plt.plot(sim.simtime / 1000, np.full(len(sim.simtime),wtot),lw=3,color='tab:red',label='W_tot')
plt.plot(sim.simtime / 1000, sim.weights_sum,lw=3,color='tab:blue',label='Sum of E weights')
plt.legend(loc='upper left')
plt.ylabel('Synaptic weigth')
plt.title(f'Weight evolution',fontweight='bold')


# PLOT WEIGHTS EVOLUTION
plt.subplot2grid((4,3),(2,0),rowspan=2,colspan=2)
# Plot group 1
for w in range(sim.stim_group_boundaries):
    plt.plot(sim.simtime/1000,sim.weights_e[w],color='tab:green',alpha=0.5)

# Plot group 1
for w in range(sim.stim_group_boundaries, len(sim.weights_e)):
    plt.plot(sim.simtime/1000,sim.weights_e[w],color='tab:purple',alpha=0.2)

# Plot averages
mean1 = np.mean(sim.weights_e[:sim.stim_group_boundaries - 1],axis=0)
plt.plot(sim.simtime / 1000, mean1, color="green", lw=4, label=f'Group 1 (c : {c1})')

mean2 = np.mean(sim.weights_e[sim.stim_group_boundaries:], axis=0)
plt.plot(sim.simtime / 1000, mean2, color="purple", lw=4, label=f'Group 2 (c : {c2})')

# Labels
plt.xlabel('Time (s)')
plt.ylabel('Synaptic weigth')
plt.legend(loc="upper left")


# PLOT WEIGHTS DISTRIBUTIONS AT DIFFERENT TIME POINTS
zero     =  int(len(sim.simtime)*1/8)
quart     = int(len(sim.simtime)*1/4)
half     = int(len(sim.simtime)*2/4)
thrq    = int(len(sim.simtime)*3/4)
full    = -1
beans = 7
for i,t in enumerate([zero, quart, thrq, full]):


    plt.subplot2grid((4,3),(i,2))


    plt.hist([x[t] for x in sim.weights_e[sim.stim_group_boundaries+1:]],
             color='purple', alpha=0.2, edgecolor='purple',bins=beans)

    plt.hist([x[t] for x in sim.weights_e[sim.stim_group_boundaries+1:]],
             color='purple', fill=False,lw=2, edgecolor='purple',
             label=f'Group 2 (c : {c2})',bins=beans)

    plt.hist([x[t] for x in sim.weights_e[:sim.stim_group_boundaries+1]],
             color='green', alpha=0.2, edgecolor='green',bins=beans)

    plt.hist([x[t] for x in sim.weights_e[:sim.stim_group_boundaries+1]],
             color='green',fill=False,lw=2, edgecolor='green',
             label=f'Group 1 (c : {c1})',bins=beans)
    plt.title(f'{int(sim.simtime[t]/1000)} seconds',fontweight='bold')
    plt.legend(loc='upper left')

    plt.ylabel('Frequency')
    if i == 3:
        plt.xlabel('Synaptic weight')


# plt.savefig(f'../results/exercise41_{fn}_{Ne}e_{Ni}i_{rate}Hz_{int(t_sim/1000)}s_wtot{wtot}_{corr}.png')

plt.show()


sim.plotVoltageTrace(show=True)

# # ------------------- PLOT ANIMATION ------------------- #
# from celluloid import Camera
#
# rcParams['figure.figsize'] = (8,6)
#
# fig = plt.figure()
# camera = Camera(fig)
#
# beans = 3
#
# for t in range(0,len(sim.simtime),50000):
#     plt.text(0.6,40,f't = {int(sim.simtime[t]/1000)} s',fontweight='bold')
#     plt.hist([x[t] for x in sim.weights_e[sim.stim_group_boundaries + 1:]],
#              color='purple', alpha=0.2, edgecolor='purple', bins=beans)
#
#     plt.hist([x[t] for x in sim.weights_e[sim.stim_group_boundaries + 1:]],
#              color='purple', fill=False, lw=2, edgecolor='purple',
#              label=f'Group 2 (c : {c2})', bins=beans)
#
#     plt.hist([x[t] for x in sim.weights_e[:sim.stim_group_boundaries + 1]],
#              color='green', alpha=0.2, edgecolor='green', bins=beans)
#
#     plt.hist([x[t] for x in sim.weights_e[:sim.stim_group_boundaries + 1]],
#              color='green', fill=False, lw=2, edgecolor='green',
#              label=f'Group 1 (c : {c1})', bins=beans)
#     camera.snap()
#
#
# plt.ylabel('Frequency')
# plt.xlabel('Synaptic weight')
#
# camera.snap()
# animation = camera.animate()
#
# # animation.save(f'../results/exercise41_{fn}_{Ne}e_{Ni}i_{rate}Hz_{int(t_sim/1000)}s_wtot{wtot}_{corr}.gif')
#
#
#
