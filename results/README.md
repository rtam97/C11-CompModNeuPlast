# Results

# Chapter 1

## 1.1 Leaky Integrate-and-Fire

In this exercise I simulated a LIF neuron receiving constant current input.

Increasing the current amplitude (from `I = 2.0 nA` to `I = 4.0 nA`) increases the firing rate of the neuron.

<p align="center">
<img src="exercise11.png" alt="exercise 11" width="500"/>
</p>


## 1.2 Periodic synaptic inputs

In the first part of this exercise, I simulated a LIF neuron for `t = 1 s` receiving a single <ins>**excitatory**</ins> synaptic input arriving **_periodically_** with a frequency of `r = 5 Hz`.

The excitatory synaptic strength was first set to `We = 0.5`, which was not enough to reach the firing threshold in the neuron.

Subsequently, the synaptic strength was increased to `We = 1.5` and to  `We = 3.0`. Only the latter was strong enough to elicit firing in the neuron.

<p align="center">
<img src="exercise12_01.png" alt="excitatory-spiking" width="500"/>
</p>

In the second part of the exercise, I added a second <ins>**inhibitory**</ins> synaptic input, arriving periodically with a frequency of `r = 3 Hz`.

Both excitatory and inhibitory synaptic weights were set to `We = Wi = 3.0`.

When the neuron received excitatory input only, it was able to emit a spike. When both excitatory and inhibitory stimuli arrived at the same time, an EPSP was observable but spike firing was inhibited.

<p align="center">
<img src="exercise12_02.png" alt="exc-inh" width="500"/>
</p>


## 1.3 Integration time-step

Euler integration was used to numerically solve the differential equations describing the evolution of neuronal activity.

The accuracy of Euler's method is dependent on the time step size `dt` chosen to move forward in time.

In this exercise, I compute the *analytical solution* for the synaptic conductance, as well as its *numerical solution(s)* with different values of `dt`.

As expected, smaller values of `dt` produce more accurate results.

<p align="center">
<img src="exercise13.png" alt="integration-time-step" width="1000"/>
</p>

## 1.4 Poisson spike trains

In this exercise I simulated a LIF neuron receiving  **_Poisson spike trains_** with constant firing rate `r = 10 Hz` (homogeneous Poisson process) as inputs to its `Ne = 10` excitatory synapses and to its `Ni = 10` inhibitory synapses. 

The neuron was simulated for `n = 50` trials, lasting `t = 10` seconds each. 

The **_inter-spike-intervals (ISI)_** were measured and their **_coefficient of variation (CV)_** was computed for each trial, together with the **_firing rate_**.   

The **distribution of ISIs** was plotted and an exponential function was fitted to it. 

The **distribution of the CVs** was also plotted and its mean calculated. 

---
One would expect that, when the Excitatory and Inhibitory inputs are balanced, the neuron would fire in accordance with the Poisson process statistics.

Therefore, one would expect the ISI to be exponentially distributed, and the CV to be close to 1. 

Furthermore, one would expect the _measured_ firing rate to be equal to around **half** of the _expected_ firing rate, given the equal contribution of excitatory and inhibitory inputs.
___


First, the synaptic strengths were set to `We = Wi = 0.5` for all synaptic inputs. 

The inter-spike-intervals were indeed exponentially distributed, however the mean CV was equal to `CV = 0.888`. 

The average firing rate was found to be lower (`r = 3.4 Hz`) than the expected value (`r = 10/2 = 5 Hz`).

These results _(figure below, graph on the left)_ suggest that the excitatory and inhibitory inputs are not balanced, with more weight given to inhibitory inputs. 
This can be explained by the fact that, in this model, the inhibitory synaptic currents are slower to decay than the 
excitatory ones, as indicted by their different time constants (`Ti = 5 ms > Te = 3 ms`). 

This implies that, if the neuron receives an excitatory and inhibitory stimuli simultaneously, and both synaptic weights are equal,
the inhibitory stimulus will have a greater (longer-lasting) effect on the overall evolution of the membrane potential.

<p align="center">
<img src="exercise14_01.png" alt="equal weights" width="400"/>
<img src="exercise14_02.png" alt="equal weights" width="400"/>
</p>

In order to explore the relation between excitatory synaptic weights and E-I balance, as well as to find the optimal `We` 
to reach this balance, a second experiment was performed _(figure above, graph on the right)_.

A series of 10 simulations (50 trials x 10 s each) were run, in which the excitatory synaptic weights for all excitatory
synapses was linearly increased from `We = 0.1` until `We = 1.0`, while the inhibitory synaptic weights were kept constant 
for all inhibitory synapses (`Wi = 0.5`).

As expected, the excitatory synaptic weight at which the highest balance was reached was slightly larger (`We = 0.6`) than 
the inhibitory one. 

In such a configuration of synaptic weights, the firing rate of the neuron was equal to `r = 7.0 Hz` and the mean CV was
 `CV = 0.901`, suggesting a greater balance of EI inputs.
 
 The fact that the firing rate is not exactly equal to `r = 5 Hz` is probably because the true optimal excitatory weight 
 actually sits between the two tested values: `0.5 < We_opt < 0.6`.
 
 ___
 
 To test the previous hypothesis, 10 more simulations were performed, this time with the excitatory synaptic weight 
 increasing linearly between `We = 0.5` until `We = 0.6` in steps of `dw = 0.01`.
 
 As expected, the optimal synaptic weight was found to be `We = 0.55`, which caused the neuron to spike with a firing rate
 of `r = 4.9 Hz` and a distribution of ISI with a coefficient of variation equal to `CV = 0.910`, the largest one achieved.
 
 <p align="center">
<img src="exercise14_03.png" alt="equal weights" width="400"/>
<img src="exercise14_04.png" alt="equal weights" width="400"/>
</p>



# Chapter 2

## 2.1 Spike-Rate Adaptation

In this exercise, I implemented a mechanism for spike-rate adaptation (SRA) in a LIF neuron. I simulated the neuron receiving a constant step current input `I_ext = 1.45` for a given amount of time. In the following graphs, one can observe the effect of SRA on the LIF neuron when it is added to the model. 

In the **first** graph the LIF is simulated **without any SRA** (`dg_sra = 0.0`) for `t_sim = 500 ms`, and a step current is applied from `stim_start = 50 ms` to `stim_end = 350 ms` . The SRA conductance is zero the entire time, and the spike frequency (inverse of the ISI for each spike) is constant all throughout the simulation

In the **second** graph the LIF is simulated **with SRA** (`dg_sra = 0.06`). The length of the simulation and the properties of the input current are the same as before. As the neuron spikes, the SRA conductance rises sharply, only to decay until the next spike is emitted. The input current is strong enough to induce sustained spiking in the neuron, so that the SRA conductance cannot decay back to `0`. Instead, the average conductance slowly grows towards its saturation point, before the step current is removed, at which point it decays back to baseline. The effect on the spiking frequency is clear: as the neuron emits more and more spikes, the inter-spike intervals get longer and longer, growing exponentially.

This is even more obvious in the **third** graph, in which the same neuron **with SRA** is simulated for a longer time period (`t_sim = 1000 ms`) and the step current is applied for longer (`750 ms`). Here, the average SRA conductance reaches its saturation point at time `t_sat ≈ 500 ms`, the same time-point at which the spiking frequency has decayed to its lowest fixed point, given the current input. After this time-point the spiking frequency will remain constant until the step current is changed.

<p align="center"> 
<img src="exercise21_01.png" alt="equal weights" width="300"/>
<img src="exercise21_02.png" alt="equal weights" width="300"/>
<img src="exercise21_03.png" alt="equal weights" width="300"/>
</p>


# Chapter 3

# Chapter 4

# Chapter 5
