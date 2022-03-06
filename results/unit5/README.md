# Results

# Chapter 5 : Short-Term Plasticity

## 5.1 Short Term Facilitation

In this exercise I implemented a mechanism for short term facilitation (STF), as described in [Tsodyks, Pawelzik & Markram (1998)](http://www.ccnss.org/ccn_2014/materials/pdf/tsodyks/Tsodyks_1998.pdf).

Here, STF is modeled phenomenologically as the result of temporary increase of neurotransmitter release probability at the pre-synaptic terminal due to calcium influx after a spike is generated in the pre-synaptic neuron. This probability is captured by the time-dependent variable `u` which obeys the following equation:

`du/dt = -u/tau_stf + U_stf * (1-u) * delta(t-t_spike)`

where `tau_stf` is the STF time constant and `U_stf` is the increase in neurotransmitter release probability due to calcium influx in the pre-synaptic neuron, which is only applied when a pre-synaptic spike occurs (ie: `t-t_spike == 0`).

Every time the pre-synaptic neuron fires an action potential, the synaptic strength between itself and the post synaptic neuron is increased as:

`w_e = w_fixed * u`

where `w_fixed` is a pre-determined increase factor (here set to 'w_fixed = 1').

___

A neuron was simulated for `t_sim = 5000 ms`, receiving `N_e = 10` excitatory inputs, of which half underwent STF, and the others did not. All synaptic weights were set to `w_e = 0.0`. The input rate was set to 'rate = 10 Hz' for all inputs.

The parameters for STF were `U_stf = 0.2`, `w_fixed = 1` and `tau_stf` was varied across experiments   

No STDP, synaptic normalization, or intrinsic plasticity were added.

No inhibitory inputs were added.

___

[Tsodyks, Pawelzik & Markram (1998)](http://www.ccnss.org/ccn_2014/materials/pdf/tsodyks/Tsodyks_1998.pdf) predicted that, for a neuron receiving periodic stimulation with `rate = r`, the neurotransmitter release probability would reach the steady-state value of: 

`U_steady = U_stf / (1 - (1 - U_stf) * exp(-1/(tau_stf*rate)))`

Therefore, I simulated the neuron receiving a `stimtype = 'periodic'` input at a rate of `r = 10 Hz`. Two simulations were performed, in which the time constant was varied between `tau_stf = 750 ms` and `tau_stf = 250 ms`.

The evolution of the neurotransmitter release probability is plotted here, together with the predicted steady-state value. Weight evolution and post-synaptic firing rates are also included. 


<p align="center"> 
  <img src="exercise51_02.png" width="450"/>
  <img src="exercise51_01.png" width="450"/>
</p>

It can be observed that in both simulations the steady state value was reached, at which point the optimal weight for keeping the desired firing rate (10 Hz) was found. A faster time constant leads to lower steady-state value, which is reached more rapidly.

___

In a second set of experiments, the same neuron was simulated with the exact same parameters, except that now the input stimuli were `poisson` distributed, without correlation between themselves.

<p align="center"> 
  <img src="exercise51_04.png" width="450"/>
  <img src="exercise51_03.png" width="450"/>
</p>

Once again, in both simulations the neurotransmitter release probability `u` reaches the steady state value, although here it varies around it due to the stochastic nature of the pre-synaptic firing.

Interestingly, a shorter time-constant leads the post-synaptic neuron to a lower firing rate than it would achieve given a larger time-constant. This is due to the fact that, with a larger time-constant, the probability of neurotransmitter release decays much more slowly, and therefore when a new pre-synaptic spike occurs, it can be summed on top of the previous one which is still decaying. 

Oppositely, if the probability decays too fast, when spikes occur in a quick sequence, the probability increases do not sum onto each other and therefore the weights are not increased as much, thus lowering the possible firing rate in the post-synaptic neuron.


## 5.2 Short Term Depression

# Go back to:

[Chapter 1 : Leaky Integrate-and-Fire](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit1/README.md)

[Chapter 2 : Adaptations in Spiking Behavior](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit2/README.md)

[Chapter 3 : Spike-Timing Dependent Plasticity](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit3/README.md)

[Chapter 4: Synaptic Homeostasis](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit4/README.md)

[Chapter 5: Short-Term Plasticity](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit5/README.md)
