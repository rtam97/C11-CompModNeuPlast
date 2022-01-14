import numpy as np
from scripts.neurons import *

# Euler integration method with spiking and voltage resetting for LIF neurons WITHOUT INPUT
def eulerLIF(neuron,dt,t_end,init,threshold,spiking,reset,parameters):

    # Define time vector and voltage vector
    time = np.linspace(0, t_end, int(t_end / dt))
    V = np.zeros(len(time))
    V[0] = init
    spikeCount = 0

    # Integration loop
    for t in range(len(time) - 1):
        if V[t] <= threshold:   # Solve equation
            V[t + 1] = V[t] + dt * neuron(V[t], **parameters)
        elif V[t] != spiking:   # Spike
            V[t + 1] = spiking
            spikeCount += 1
        else:                   # Reset
            V[t + 1] = reset
    return V, time, spikeCount


# Euler integration for LIF with synaptic inputs (E : I)
def eulerLIFinputs(neuron,synapse,
                   dt, t_end,
                   init, threshold,spiking,reset,
                   E_leak, tau_m,
                   E_ex, tau_e, w_ex, f_ex, t_ex,
                   E_in, tau_i, w_in, f_in, t_in):

    # Vectors
    time = np.linspace(0, t_end, int(t_end / dt))
    V = np.zeros(len(time))
    V[0] = init
    g_ex = np.zeros(len(time))
    g_ex[0] = 0
    g_in = np.zeros(len(time))
    g_in[0] = 0

    # Excitatory stimulus times
    ts = generatePeriodicStimulus(f_ex, t_ex, time, dt)

    # Inhibitory stimulus times
    tp = generatePeriodicStimulus(f_in, t_in, time, dt)

    # Spike counter
    spikeCount = 0

    # Integration loop
    for t in range(len(time) - 1):
        if V[t] <= threshold:  # Solve equations

            g_ex[t + 1] = g_ex[t] + dt * synapse(g_ex[t], time[t], ts, tau_e, w_ex)
            g_in[t + 1] = g_in[t] + dt * synapse(g_in[t], time[t], tp, tau_i, w_in)
            V[t + 1] = V[t] + dt * neuron(V[t], E_leak, tau_m, g_ex[t + 1], E_ex, g_in[t + 1], E_in)

        elif V[t] != spiking:  # Spike

            g_ex[t + 1] = g_ex[t] + dt * synapse(g_ex[t], time[t], ts, tau_e, w_ex)
            g_in[t + 1] = g_in[t] + dt * synapse(g_in[t], time[t], tp, tau_i, w_in)
            V[t + 1] = spiking
            spikeCount += 1

        else:  # Reset
            g_ex[t + 1] = g_ex[t] + dt * synapse(g_ex[t], time[t], ts, tau_e, w_ex)
            g_in[t + 1] = g_in[t] + dt * synapse(g_in[t], time[t], tp, tau_i, w_in)
            V[t + 1] = reset
    return V, g_ex, g_in, time, spikeCount
