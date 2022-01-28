from neurosim.neurosim import Neuron
from neurosim.neurosim import Stimulus
from neurosim.neurosim import Simulation
from matplotlib import pyplot as plt

# --------------------- LIF NEURON ---------------------

# Instantiate LIF neuron with default parameters
lif = Neuron(type='LIF')

# --------------------- CONSTANT CURRENT : 2.0 nA ---------------------

# Create a constant stimulus with current 2.0 nA
stim1 = Stimulus(neuron=lif, stim_type='constant', I_ext=2.0)

# Instantiate a simulation object
sim1 = Simulation(neuron=lif,stimulus=stim1,t_sim=100)

# Run simulation
sim1.simulate()

# Save parameters used for simulation
sim1.saveInputParameters('param1.txt')

# --------------------- CONSTANT CURRENT : 4.0 nA ---------------------

# Create a constant stimulus with current 2.0 nA
stim2 = Stimulus(neuron=lif, stim_type='constant', I_ext=4.0)

# Instantiate a simulation object
sim2 = Simulation(neuron=lif,stimulus=stim2,t_sim=100)

# Run simulation
sim2.simulate()

# save parameters used for simulation
sim2.saveInputParameters('param2.txt')

# --------------------- PLOT ---------------------

plt.subplot2grid((2,2),(0,0))
Vplot = sim1.plotVoltageTrace(label="I_ext = 2.0 nA")
plt.subplot2grid((2,2),(1,0))
Vplot2 = sim2.plotVoltageTrace(label="I_ext = 4.0 nA")


# Plot Firing rates
plt.subplot2grid((2,2),(0,1),rowspan=2)
plt.bar(height=[sim1.firingRate,sim2.firingRate],
        x=[1,2],
        fill=0,
        edgecolor=['tab:blue','tab:orange'])
plt.text(0+0.9,sim1.firingRate-5,sim1.firingRate,color='tab:blue')
plt.text(1+0.9,sim2.firingRate-5,sim2.firingRate,color='tab:orange')
plt.xticks([1,2],[2.0,4.0])
plt.xlabel("Injected Current (nA)")
plt.ylabel("Firing Rate (Hz)")
# plt.savefig('../results/exercise11.png')
plt.show()

