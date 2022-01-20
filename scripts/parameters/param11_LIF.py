# Neuron parameters
E_leak = -60         # Reversal potential       (mV)
tau_m = 20           # Membrane time constant   (ms)    --> Rm*Cm [implicit]
R_m = 10             # Membrane resistance      (Mohm)
currents = [2.0,4.0] # Injected currents        (nA)

# Spiking parameters
V_spike = 0         # Spike amplitude       (mV)
V_reset = -70       # Reset potential       (mV)
V_theta = -50       # Threshold potential   (mV)
V_init = V_reset    # Initial potential     (mV)


# Euler integration parameters
t_end = 100        # Simulation time       (ms)
dt = 0.1            # Integration time step (ms)
