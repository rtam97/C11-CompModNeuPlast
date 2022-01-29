from neurosim.neurosim import Neuron, Stimulus, Simulation
from matplotlib import pyplot as plt
import matplotlib as mlp
mlp.rcParams['figure.figsize'] = (9,5)


# Create neuron with refractory period and BALANCED synaptic inputs
lif_refperiod = Neuron(type='lif',
                       N_exc=10, w_exc= 0.55,
                       N_inh=10, w_inh=0.5,
                       ref=1.2, E_rp=-70, tau_rp=50)

# Create poisson stimulus
poisson_stim = Stimulus(stim_type='poisson',
                        rate_exc=10,rate_inh=10,
                        t_sim = 10000,dt=0.1,
                        neuron=lif_refperiod)

# Create simulation with given input and neuron
sim = Simulation(neuron=lif_refperiod,
                 stimulus=poisson_stim)

# Run multi-trial simulation
sim.simulate(trials=50)

# Plots
plt.subplot(1,2,1)
sim.plotISIdist(expfit=True)
plt.subplot(1,2,2)
sim.plotCVdist()
title = 'We: %s || ' \
        'Wi: %s || ' \
        'Trials: %s x %s s || ' \
        'Rate (E-I): %s - %s Hz || ' \
        'Firing rate: %s Hz'\
            %(lif_refperiod.w_exc[0],lif_refperiod.w_inh[0],
              sim.trials, sim.t_sim/1000,
              poisson_stim.rate_exc, poisson_stim.rate_inh,
              sim.meanFR)
plt.suptitle(title,fontweight='bold')
plt.show()