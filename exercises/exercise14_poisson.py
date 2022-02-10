from neurosim.neurosim import Neuron, Stimulus, Simulation
from matplotlib import pyplot as plt
import matplotlib as mlp
from time import time as sw

mlp.rcParams['figure.figsize'] = (9,9)

#%%
# ------------------ FIGURE 1 ------------------- #
#    Poisson input with equal synaptic weight     #
# ----------------------------------------------- #

# Create neuron with 10 E and 10 I synapses with equal weights
lif = Neuron(type="lif",
             N_exc=10, w_exc=0.5,
             N_inh=10, w_inh=0.5)

# Create poisson stimulus
stim = Stimulus(stim_type='poisson', neuron=lif,
                rate_exc=10, rate_inh=10,
                t_sim=10000, dt=0.1)

# Create simulation object
sim = Simulation(neuron=lif,stimulus=stim,dt=0.1)


# Run multi-trial situation
sim.simulate(trials=5)


# Plot voltage (last trial only)
plt.subplot2grid((2,2),(0,0),colspan=2)
sim.plotVoltageTrace()
title = 'We: %s || ' \
        'Wi: %s || ' \
        'Trials: %s x %s s || ' \
        'Rate (E-I): %s - %s Hz || ' \
        'Firing rate: %s Hz'\
            %(lif.w_exc[0],lif.w_inh[0],
              sim.trials, sim.t_sim/1000,
              stim.rate_exc, stim.rate_inh,
              sim.meanFR)
plt.title(title,fontweight='bold')

# Plot inter-spike interval distrbution with exponential fit
plt.subplot2grid((2,2),(1,0))
sim.plotISIdist(expfit=True)

# Plot distribution of CVs across trials
plt.subplot2grid((2,2),(1,1))
sim.plotCVdist()

from numpy import isnan

sim.CV = sim.CV[~isnan(sim.CV)]
sim.CV = sim.CV[sim.CV != 0.]
print( sim.CV )


plt.show()

# %%
# ----------------------- FIGURES 2-3-4 ------------------------ #
#       Finiding synaptic weights for maximised EI balance       #
# -------------------------------------------------------------- #
import numpy as np

# Generate array of weights to test
weights = np.round(np.linspace(0.1,1.0,10),2)

CV = []
SC = []
FR = []

# Iterate through all the weights being tested
for i,w_e in enumerate(weights):

    print(i+1, ' out of ', len(weights))

    # Create LIF neuron with 10 E and 10 I synapses
    lif = Neuron(type="lif",
                 N_exc=10, w_exc=w_e,
                 N_inh=10, w_inh=0.5)

    # Create poisson stimulus
    stim = Stimulus(stim_type='poisson', neuron=lif,
                    rate_exc=10, rate_inh=10,
                    t_sim=10000, dt=0.1)

    # Create simulation object
    sim = Simulation(neuron=lif, stimulus=stim, dt=0.1)

    # Run multi-trial situations
    sim.simulate(trials=50)

    # Save CV, SC, and FR data
    CV.append(sim.meanCV)
    SC.append(sim.meanSpikes)
    FR.append(sim.meanFR)

    # region # -------- vvv SINGLE-TRIAL PLOTS vvv -------- #
    # Plot voltage (last trial only)
    plt.subplot2grid((2, 2), (0, 0), colspan=2)
    sim.plotVoltageTrace()
    title = 'We: %s || Wi: %s || Trials: %s x %s s || Rate (E-I): %s - %s Hz || Firing rate: %s' \
            % (round(lif.w_exc[0],2), round(lif.w_inh[0],2),
               sim.trials, sim.t_sim / 1000,
               stim.rate_exc,
               stim.rate_inh,
               sim.meanFR)
    plt.title(title, fontweight='bold')

    # Plot inter-spike interval distrbution with exponential fit
    plt.subplot2grid((2, 2), (1, 0))
    try:
        sim.plotISIdist(expfit=True)
    except Exception as e:
        plt.text(0.5,0.5,'No spikes generated',ha='center', va='center',fontsize=24,fontweight='bold')

    # Plot distribution of CVs across trials
    plt.subplot2grid((2, 2), (1, 1))
    try:
        sim.plotCVdist()
    except Exception as e:
        plt.text(0.5, 0.5, 'No spikes generated', ha='center', va='center', fontsize=24, fontweight='bold')
    # endregion

    # Save single-trial-plot for the current weight
    plt.savefig('../results/temp/weight%s.png'%(int(w_e*100)))
    # plt.show()


# Plot overall CVs and Firing rate as a function of W_ex
mlp.rcParams['figure.figsize'] = (7,7)
fig,ax = plt.subplots()

# Coefficient of Variation
ax.plot(weights,CV,label='CV',
         linewidth=2,color='tab:green',
         marker='D',markersize=8,markerfacecolor='green',markeredgecolor='green')
ax.set_ylabel('CV',color='green',fontsize=14,fontweight='bold')
ax.set_xlabel('Excitatory synaptic weights')
plt.legend(loc='upper left')

# Line through the MAX(CV)
BW = weights[np.where(CV==np.max(CV))[0]]
plt.plot([BW]*100,np.linspace(np.min(CV)-0.05,np.max(CV)+0.05,100),'--',linewidth=3,color='tab:red')

# Firing rate on same plot
ax2 = ax.twinx()
ax2.plot(weights,FR,label='Firing rate',
         linewidth=2, color='tab:purple',
         marker='v', markersize=8, markerfacecolor='purple', markeredgecolor='purple')
ax2.set_ylabel('Firing rate (Hz)',color='purple',fontsize=14,fontweight='bold')
ax2.set_xticks(weights)
plt.legend(loc='upper right')

plt.title('CV and Spikes given W_ex',fontweight='bold')
plt.savefig('results/temp/final_exercise14_03.png')
plt.show()
