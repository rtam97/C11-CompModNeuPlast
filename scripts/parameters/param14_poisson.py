# Neuron parameters
E_leak = -60         # Leak reversal potential  (mV)
tau_m = 20           # Membrane time constant   (ms)    --> Rm*Cm [implicit]

# Excitatory input parameters
E_ex    = 0     # Reversal potential    (mV)
tau_e   = 3     # Time constant         (ms)
w_ex    = 0.5   # Synapse strength      (--)
t_ex    = 1     # Stimulus duration     (ms)
# Poisson spiking
f_ex    = 10  # Firing rate           (Hz)

# Inhibitory input parameters
E_in    = -80   # Reversal potential    (mV)
tau_i   = 5     # Time constant         (ms)
w_in    = 0.5   # Synapse strength      (--)
t_in    = 1     # Stimulus duration     (ms)
# Poisson spiking
f_in    = 10   # Firing rate           (Hz)

# Reset potential parameters
V_spike = 0         # Spike amplitude       (mV)
V_reset = -70       # Reset potential       (mV)
V_theta = -50       # Threshold potential   (mV)
V_init = V_reset    # Initial potential     (mV)

# Euler integration parameters
t_end = 50000      # Simulation time       (ms)
dt = 1            # Integration time step (ms)

