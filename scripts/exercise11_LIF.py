# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                   C11 - COMPUTATIONAL MODELING OF NEURONAL PLASTICITY                               #
#                                                   EXERCISE 1.1                                                      #
#                                                  RUBEN  TAMMARO                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
from scripts.solvers import eulerOdeSpike
from scripts.neurons import LIF

# ----------------------------------------------------- PARAMETERS -----------------------------------------------------


# Neuron parameters
E_leak = -60         # Reversal potential       (mV)
tau_m = 20           # Membrane time constant   (ms)    --> Rm*Cm [implicit]
R_m = 10             # Membrane resistance      (Mohm)
currents = [2.0,4.0] # Injected currents        (nA)

# Euler integration parameters
t_end = 100         # Simulation time       (ms)
dt = 0.1            # Integration time step (ms)
V_spike = 0         # Spike amplitude       (mV)
V_reset = -70       # Reset potential       (mV)
V_theta = -50       # Threshold potential   (mV)
V_init = V_reset    # Initial potential     (mV)


# ----------------------------------------------------- SIMULATION -----------------------------------------------------

# Result variables
spikeCounts = []
firingRates = []
potentials  = []

# Simulate for multiple current inputs
for I_ext in currents:

    # Neuron parameters dictionary
    # (keys = function input parameters)
    params = {
        'E' : E_leak,
        'R' : R_m,
        'I' : I_ext,
        'tau' : tau_m
    }

    # Integration routine
    (V,time,sc) = eulerOdeSpike(neuron      =   LIF,
                                dt          =   dt,
                                t_end       =   t_end,
                                init        =   V_init,
                                threshold   =   V_theta,
                                spiking     =   V_spike,
                                reset       =   V_reset,
                                parameters  =   params)

    # Saving variables
    potentials.append(V)
    spikeCounts.append(sc)
    firingRates.append(sc/t_end*1000)


# ------------------------------------------------------- PLOTS --------------------------------------------------------

# Plot voltage time series | input current 2.0
plt.subplot(2,2,1)
plt.plot(time, potentials[0], label="I_ext = %s nA" % (currents[0]))
plt.legend(loc='upper left',fontsize='x-small')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")

# Plot voltage time series | input current 4.0
plt.subplot(2,2,3)
plt.plot(time, potentials[1], label="I_ext = %s nA" % (currents[1]))
plt.legend(loc='upper left',fontsize='x-small')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")

# Plot voltage time series | all input currents
plt.subplot(2,2,2)
for i,pot in enumerate(potentials):
    plt.plot(time,pot,label="I_ext = %s nA"%(currents[i]))
plt.legend(loc='upper left',fontsize='x-small')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")

# Plot Firing rates
plt.subplot(2,2,4)
plt.bar(height=firingRates,
        x=list(range(len(currents))),
        fill=0,
        edgecolor=['tab:blue','tab:orange'])
plt.xticks(list(range(len(currents))),currents)
plt.xlabel("Injected Current (nA)")
plt.ylabel("Firing Rate (Hz)")
plt.show()