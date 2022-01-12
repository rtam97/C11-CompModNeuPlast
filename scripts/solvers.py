import numpy as np

# Euler integration method with spiking and voltage resetting
# Should allow for multiple neuron types to be simulated
def eulerOdeSpike(neuron,dt,t_end,init,threshold,spiking,reset,parameters):

    # Define time vector and voltage vector
    time = np.linspace(0, t_end, int(t_end / dt))
    V = np.zeros(len(time))
    V[0] = init
    spikeCount = 0

    # Integration loop
    for t in range(len(time) - 1):
        if V[t] <= threshold:   # Load capacitor
            V[t + 1] = V[t] + dt * neuron(V[t], **parameters)
        elif V[t] != spiking:   # Spike
            V[t + 1] = spiking
            spikeCount += 1
        else:                   # Reset
            V[t + 1] = reset
    return V, time, spikeCount