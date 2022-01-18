import numpy as np


# --------------------------------------------- NEURONS --------------------------------------------- #

# Leaky integrate-and-fire (LIF) neuron
def LIF(V,E,R,I,tau):
    return (E - V+R*I)/tau

# Leaky integrate-and-fire (LIF) neuron + EI inputs
def LIF_inputs(V, g_ex, g_in, E_leak, tau_m, E_ex, E_in):
    I_ex = g_ex*(E_ex - V)
    I_in = g_in*(E_in - V)
    return (E_leak - V + np.sum(I_ex) + np.sum(I_in))/tau_m


# Conductance of EI inputs
def g_synapse(g,t,t_stim,tau,w):
    return -g / tau + w * dirac(t, t_stim)


# --------------------------------------------- INPUTS --------------------------------------------- #


# Dirac's delta function
def dirac(t,t_inputs):
    delta = 0
    if t in t_inputs:
        delta = 1

    return delta


# Generates the spike times at which an input spike is generated, sampling from Poisson distribution
# Algorithm from 'Theoretical Neuroscience' by Dayan & Abbott (2001), Page 30: The Poisson Spike Generator
def generatePoissonStimulus(rate,time,dt):
    spike_times = [t for t in time if rate*(dt/1000) > np.random.random()]
    return spike_times


# Defines the timepoints at which to induce a periodic stimulus based on desired frequency of given length in ms
def generatePeriodicStimulus(frequency,stim_length,time,dt):
    stim_idx = []
    stim = []
    if frequency != 0 :
        ISI = 1000/frequency

        # Find time indexes of stimulus start for a defined length stimulus
        stim_start_idx = [t for t in range(len(time) - 1)
                          if round(np.mod(round(time[t],1),round(ISI,1))) == 0.0
                          and round(time[t]) != 0.0]

        # Reject simulus start points falling within the previous stimulus length
        stim_start_idx = [x for i,x in enumerate(stim_start_idx)
                          if x - stim_start_idx[i-1] > int(stim_length / dt)
                          and i != 0]

        # Find time indexes of stimulus end (depends on stim_length)
        stim_end_idx = [x + int(stim_length / dt) for x in stim_start_idx]

        # Create range of indexes for stimulus duration
        for i,x in enumerate(stim_start_idx):
            stim_idx.append(list(range(stim_start_idx[i],stim_end_idx[i])))

        # Find exact times of indexes in time vector
        for s in stim_idx:
            for x in s:
                try:
                    stim.append(time[x])
                except Exception as e:
                    pass

    return stim
