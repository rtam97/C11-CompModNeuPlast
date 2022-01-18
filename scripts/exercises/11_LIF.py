# --------------------------------------------------------- #
# C11 - COMPUTATIONAL MODELING OF SYNAPTIC PLASTICITY       #
# EXERCISE 1.1 - LEAKY INTEGRATE AND FIRE                   #
# RUBEN TAMMARO                                             #
# --------------------------------------------------------- #

import matplotlib.pyplot as plt
from scripts.solvers import eulerLIF as solve
from scripts.neurons import LIF

# ----------------------------------------------------- PARAMETERS -----------------------------------------------------

from scripts.parameters.param11_LIF import *

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
    (V,time,sc) = solve(neuron      =   LIF,
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
plt.subplot2grid((2,2),(0,0))
plt.plot(time, potentials[0], label="I_ext = %s nA" % (currents[0]))
plt.legend(loc='upper left',fontsize='x-small')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")

# Plot voltage time series | input current 4.0
plt.subplot2grid((2,2),(1,0))
plt.plot(time, potentials[1], label="I_ext = %s nA" % (currents[1]),color='tab:orange')
plt.legend(loc='upper left',fontsize='x-small')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")

# Plot Firing rates
plt.subplot2grid((2,2),(0,1),rowspan=2)
plt.bar(height=firingRates,
        x=list(range(len(currents))),
        fill=0,
        edgecolor=['tab:blue','tab:orange'])
plt.text(0-0.1,firingRates[0]-5,firingRates[0],color='tab:blue')
plt.text(1-0.1,firingRates[1]-5,firingRates[1],color='tab:orange')
plt.xticks(list(range(len(currents))),currents)
plt.xlabel("Injected Current (nA)")
plt.ylabel("Firing Rate (Hz)")
plt.show()