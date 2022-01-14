import numpy as np


# --------------------------------------------- NEURONS --------------------------------------------- #

# Leaky integrate-and-fire (LIF) neuron
def LIF(V,E,R,I,tau):
    return (E - V+R*I)/tau

# Leaky integrate-and-fire (LIF) neuron + EI inputs
def LIF_inputs(V, g_ex, g_in, E_leak, tau_m, E_ex, E_in):
    I_ex = g_ex*(E_ex - V)
    I_in = g_in*(E_in - V)
    return (E_leak - V + I_ex + I_in)/tau_m


# Conductance of EI inputs
def g_synapse(g,t,t_stim,tau,w):
    return -g / tau + w * dirac(t, t_stim)


# --------------------------------------------- INPUTS --------------------------------------------- #


# Dirac's delta function
def dirac(t,t_inputs):
    delta = 0
    for tstim in t_inputs:
        if t == tstim :
            delta += 1
    return delta

# Defines the timepoints at which to induce a periodic stimulus based on desired frequency of given length in ms
def generatePeriodicStimulus(frequency,stim_length,time,dt):
    stim_idx = []
    stim = []
    if frequency != 0 :
        ISI = 1000/frequency

        # Find time indexes of stimulus start (into integer)
        stim_start_idx = [np.where(time == t) for t in time if round(np.mod(round(t,1),round(ISI,1)),1) == 0.0 and round(t) != 0.0]
        stim_start_idx = [int(x[0]) for x in stim_start_idx]

        # Find time indexes of stimulus end (depends on stim_length)
        stim_end_idx = [x + int(stim_length / dt) for x in stim_start_idx]

        # Create range of indexes for stimulus duration
        for i,x in enumerate(stim_start_idx):
            stim_idx.append(list(range(stim_start_idx[i],stim_end_idx[i])))

        # Find exact times of indexes in time vector
        for s in stim_idx:
            for x in s:
                stim.append(time[x])

    return stim
