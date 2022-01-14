# --------------------------------------------------------- #
# C11 - COMPUTATIONAL MODELING OF SYNAPTIC PLASTICITY       #
# EXERCISE 1.2 - SYNAPTIC INPUTS                            #
# RUBEN TAMMARO                                             #
# --------------------------------------------------------- #

import matplotlib.pyplot as plt
from scripts.solvers import eulerLIFinputs as solve
from scripts.neurons import *



# ----------------------------------------------------- PARAMETERS -----------------------------------------------------

from scripts.parameters.param12_LIF_inputs import *

# Neuron parameters dictionary
# (keys = function input parameters)

params = {
    'E_leak': E_leak,
    'tau_m': tau_m,

    'E_ex': E_ex,
    'w_ex': w_ex,
    'tau_e': tau_e,
    'f_ex' : f_ex,
    't_ex' : t_ex,

    'E_in': E_in,
    'w_in': w_in,
    'tau_i': tau_i,
    'f_in' : f_in,
    't_in' : t_in
}



# ----------------------------------------------------- SIMULATION -----------------------------------------------------

# Result variables
spikeCounts = []
firingRates = []
potentials = []


# Euler integration parameters
t_end = 1000        # Simulation time       (ms)
dt = 0.1            # Integration time step (ms)


# Euler integration
(V,g_ex,g_in,time,sc) = solve(neuron=LIF_inputs,
                              synapse=g_synapse,
                              dt=dt,
                              t_end=t_end,
                              init=V_reset,
                              threshold=V_theta,
                              spiking=V_spike,
                              reset=V_reset,
                              **params)


# ------------------------------------------------------- PLOTS --------------------------------------------------------



# Plot voltage time series
plt.subplot(2,1,1)
plt.plot(time, V,label="We = %s\nWi = %s"%(w_ex,w_in))
plt.legend(loc='upper left',fontsize='x-small')
plt.ylabel("Membrane Potential (mV)")

# Plot EI conductances
plt.subplot(2,1,2)
plt.plot(time, g_ex, color='g', label='Excitatory')
plt.plot(time, -g_in, color='m', label='Inhibitory')
plt.legend(loc='upper left',fontsize='x-small')
plt.xlabel('Time (ms)')
plt.ylabel('Conductance (mS)')
plt.show()



