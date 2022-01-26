# Results

## Exercise 1.1 - Leaky IF

In this exercise I simulated a LIF neuron receiving constant current input. 

Increasing the current amplitude increases the firing rate of the neuron.

<p align="center">
<img src="11_LIF.png" alt="drawing" width="500"/>
</p>

The parameters used were:
```python
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
```



## Exercise 1.2 - Leaky IF with periodic synaptic inputs

In this exercise, I first simulated a LIF neuron receiving a single excitatory synaptic input.

By increasing the strength of the synapse, we can get the neuron to reach the threshold and fire.

<p align="center">
<img src="12_LIF-inputs_strength_ramp.png" alt="drawing" width="500"/>
</p>

Parameters used:
```python
# Neuron parameters
E_leak = -60         # Leak reversal potential  (mV)
tau_m = 20           # Membrane time constant   (ms)    --> Rm*Cm [implicit]

# Excitatory input parameters
E_ex    = 0     # Reversal potential    (mV)
tau_e   = 3     # Time constant         (ms)
w_ex    = 3.0   # Synapse strength      (--)
f_ex    = 6     # Input frequency       (Hz)
t_ex    = 1     # Stimulus duration     (ms)
```

Then, I simulated a LIF neuron receiving two synaptic inputs, an inhibitory and an excitatory one.

If the two synapses have the same strength, and if the neuron receives simultaneous inputs from both, firing is inhibited.

<p align="center">
<img src="12_LIF-inputs_periodic_e6hz_i3hz.png" alt="drawing" width="500"/>
</p>

Parameters used:
```python
# Neuron parameters
E_leak = -60         # Leak reversal potential  (mV)
tau_m = 20           # Membrane time constant   (ms)    --> Rm*Cm [implicit]

# Excitatory input parameters
E_ex    = 0     # Reversal potential    (mV)
tau_e   = 3     # Time constant         (ms)
w_ex    = 3.0   # Synapse strength      (--)
f_ex    = 6     # Input frequency       (Hz)
t_ex    = 1     # Stimulus duration     (ms)

# Inhibitory input parameters
E_in    = -80   # Reversal potential    (mV)
tau_i   = 5     # Time constant         (ms)
w_in    = 3.0   # Synapse strength      (--)
f_in    = 3    # Input frequency       (Hz)
t_in    = 1     # Stimulus duration     (ms)

# Reset potential parameters
V_spike = 0         # Spike amplitude       (mV)
V_reset = -70       # Reset potential       (mV)
V_theta = -50       # Threshold potential   (mV)
V_init = V_reset    # Initial potential     (mV)
```

⚠️ Note that here the synaptic conductance reaches values of maximum +-2 mS , while in the lecture notes the range is +100 -50. Maybe the figure was created with different parameters? ⚠️

## Exercise 1.3 - Integration Time Step

Euler integration is used to numerically solve the differential equations describing the evolution of neuronal activity.

The accuracy of Euler's method is dependent on the time step size `dt` chosen to move forward in time.

In this exercise, I find the *analytical solution* for the synaptic conductance, as well as the *numerical solution(s)* with different values of `dt`.

Smaller values of `dt` produce more accurate results.

<p align="center">
<img src="13_TimeStep.png" alt="drawing" width="800"/>
</p>


## Exercise 1.4 - Poisson spike train

For this exercise, a function generating (homogeneous) Poisson spike trains was created and was used as an input for the LIF neuron.

Furthermore, the LIF neuron was expanded to allow for an arbitrary number of excitatory and inhibitory neurons.

The neuron was simulated with 10 excitatory and 10 inhibitory input synapses, initially with equal strength (**We = Wi = 0.5**), and an expected firing rate of **10 Hz**. In order to comply with such a firing rate, the time step was increased to 1 ms.

50 simulations (trials) of 50 seconds each were performed in order to obtain enough data points to compute statistics for the *inter-spike-intervals* (**ISI**) the *coefficient of variation* (**CV**). 

Histograms for both ISI and CV are plotted, together with the voltage time series of the last 50s trial. 
<p align="center">
  <img src="14_poisson_0.png" alt="drawing" width="50%"/>
</p>


Parameters used:
```python
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

```


Secondly, a series of 10 simulation experiments (like the one described above - 50x50s) were performed with excitatory synaptic strength linearly increasing from 0.5 to 1.5 in order to find out at which point the synaptic inputs are balanced between excitation and inhibition.

Apparently, increasing the excitatory synaptic strength by 0.1 already renders the inputs more inbalanced than before. Possibly a smaller increase (eg 0.5) might instead produce an increase in input balance, after which it will start to decline, as we observe.

<p align="center">
  <img src="14_poisson_1.png" alt="drawing" width="40%"/>
  <img src="14_poisson_CVISI.png" alt="drawing" width="40%"/>
</p>
