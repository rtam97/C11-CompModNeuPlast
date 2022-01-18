import numpy as np
from scripts.neurons import *
from time import time as stopwatch


# Euler integration method with spiking and voltage resetting for LIF neurons WITHOUT INPUT
# FOR NOW LIF W/o INPUTS ONLY WORKS WITH THIS
# IN THE FUTURE I WILL PUT EVERYTHIN IN A SINGLE EULER INTEGRATION FUNCTION
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


# Euler method to integrate a LIF neuron with arbitrary number of
# Excitatory and inhibitory synapses
def eulerNeuron(neuron,synapse,         # Functions describing the evolution of voltages and conductances
       Ne,Ni,                           # Number of Ex and In synapses
       dt, t_end,                       # Time-step and simulation length
       init, threshold,spiking,reset,   # Spiking parameters
       stim_type,                       # Stimulus type
       E_leak, tau_m,                   # Membrane properties
       E_ex, tau_e, w_ex, f_ex, t_ex,   # Excitatory synapse properties
       E_in, tau_i, w_in, f_in, t_in):  # Inhibitory synapse properies


    # Time vector
    time = np.linspace(0, t_end, int(t_end / dt))

    # Voltage vector
    V = np.zeros(len(time))
    V[0] = init

    # Synaptic conductances vectors (EXCITATORY)
    if Ne > 0 :
        g_ex = np.zeros((Ne,(len(time))))
    else:
        g_ex = np.zeros(len(time))


    # Synaptic conductances vectors (INHIBITORY)
    if Ni > 0:
        g_in = np.zeros((Ni,(len(time))))
    else:
        g_in = np.zeros(len(time))


    # Spike time inputs
    if stim_type == 'poisson':
        st_ex = [generatePoissonStimulus(f_ex,time,dt) for st in range(len(g_ex))]
        st_in = [generatePoissonStimulus(f_in,time,dt) for st in range(len(g_in))]

    elif stim_type == 'periodic':
        st_ex = generatePeriodicStimulus(f_ex, t_ex, time, dt)
        st_in = generatePeriodicStimulus(f_in, t_in, time, dt)
    else:
        st_ex = 0
        st_in = 0


    # Spike counter
    spikeCount = 0
    spike_times = []

    times = []


    for t in range(len(time) - 1):

        # if np.mod(t,int(len(time)/10)) == 0:
        #     print('\t%s / %s'%(t,len(time)))

        # Update EXCITATORY synaptic conductances
        if Ne > 1:
            for m in range(Ne):
                g_ex[m][t + 1] = g_ex[m][t] + dt * synapse(g_ex[m][t], time[t], st_ex[m], tau_e, w_ex)
        elif Ne == 1:
            g_ex[0][t + 1] = g_ex[0][t] + dt * synapse(g_ex[0][t], time[t], st_ex, tau_e, w_ex)

        # Update INHIBITORY synaptic conductances
        if Ni > 1:
            for n in range(Ni):
                g_in[n][t + 1] = g_in[n][t] + dt * synapse(g_in[n][t], time[t], st_in[n], tau_i, w_in)
        elif Ni == 1:
            g_in[0][t + 1] = g_in[0][t] + dt * synapse(g_in[0][t], time[t], st_in, tau_i, w_in)

        # Check threshold potential
        if V[t] <= threshold:   # Update VOLTAGE
            V[t + 1] = V[t] + dt * neuron(V[t], g_ex[:,t + 1], g_in[:,t + 1], E_leak, tau_m, E_ex, E_in)
        elif V[t] != spiking:   # Spike
            V[t + 1] = spiking
            spikeCount += 1
            spike_times.append(time[t])
        else:                   # Reset
            V[t + 1] = reset


    return V, np.squeeze(g_ex), np.squeeze(g_in), time, spikeCount,spike_times