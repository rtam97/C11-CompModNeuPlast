# Results

# Chapter 3

## 3.1 Spike-Timing Dependent Plasticity

In this exercise I implemented a mechanism for Spike-Timing Dependent Plasticity (STDP, in which the weight of each synaptic input was governed by the following differential equation:

`dw[t]/dt  = A_ltp*X[t]*d[t-t_pre] + A_ltd*Y[t]*d[t-t_post]`

where `X[t]` and `Y[t]` are the spike traces for the pre- and post-synaptic neurons, respectively, as described in [Morrison et al., 2008](https://doi.org/10.1007/s00422-008-0233-1), and `d[*]` is the Dirac's delta function. 

STDP was added to a traditional LIF neuron, receiving `N_exc = 2` excitatory synapses and no inhibitory ones. The two excitatory synapses had equal initial weights (`w_exc = 1.0`). The LTP amplitude was set to `A_ltp = 0.1` and the LTD amplitude was set to `A_ltd = -0.05`. Time constants for LTP and LTD were set to `tau_ltp = 17 ms` and `tau_ltd = 34 ms`.

The neuron was simulated for `t_sim = 15` seconds, while receiving poisson distributed inputs from both pre-synaptic neurons, with expected firing rate `rate_exc = 5 Hz` for both.


<p align="center"> 
<img src="exercise31_01.png" alt="equal weights" width="320"/>
<img src="exercise31_02.png" alt="equal weights" width="320"/>
<img src="exercise31_03.png" alt="equal weights" width="320"/>
</p>

In a second experiment the expected firing rate of one of the synapses was increased to `rate_exc[0] = 8 Hz` while the other one remained at `rate_exc[1] = 5 Hz`. 

<p align="center"> 
<img src="exercise31_04.png" alt="equal weights" width="320"/>
<img src="exercise31_05.png" alt="equal weights" width="320"/>
<img src="exercise31_06.png" alt="equal weights" width="320"/>
</p>

## 3.2 Correlated Spike Trains

In this exercise I implemented a function to generate time-correlated spike trains. The procedure follows the algorithm presented _method 2_ of [Brette, 2008](http://romainbrette.fr/WordPress3/wp-content/uploads/2014/06/Brette2008NC.pdf). Briefly, a Poisson-distributed `source_train` is generated as [before](#3.2-Correlated-Spike-Trains). Next, each spike is copied to a `new_train`, with a probability of `p = sqrt(c)`, where `c` is the desired correlation between spike trains. In order to make up for the loss of firing rate caused by the thinning, another `noise_train` is generated with rate `r = r_source*(1-p)`, and it is combined with the `new_train` to generate the `final_train`. The method described generates instantaneous (`'inst'`) correlation between spike trains. I also implemented the possibility to generate exponentially (`'exp'`) correlated spike trains. To achieve this each new spike was jittered by an exponentially distributed random amount.

I also implemented a method to compute and plot the cross-correlogram between 2 or more spike trains. It follows the method described in [Dayan and Abbot, 2007](http://www.gatsby.ucl.ac.uk/~lmate/biblio/dayanabbott.pdf).

I created two groups of 10 _instantaneously_ correlated Poisson spike trains, with correlation `c = 0.3`, firing rate `r = 10 Hz` and duration `t_stim = 10 s`. The cross-correlograms `'within'` each group show that, indeed, for the majority of the spike pairs in the 10 spike trains, the lag between them is equal to zero, indicating that they are firing synchronously. When the cross-correlation is however computed `'between'` Group 1 and Group 2, no correlation was found, as evidenced by the lack of a peak at zero-lag.

<p align="center"> 
<img src="exercise32_02.png" alt="instantaneous-correlation" width="1000"/>
</p>

When the same experiment was repeated with 2 groups of 10 _exponentially_ correlated spike trains, similar results occurred: Correlation within groups showed a peak at the zero-lag, which however did not decline sharply but which decayed exponentially, as expected. Once again, correlation between-groups was absent.

<p align="center"> 
<img src="exercise32_01.png" alt="exponential-correlation" width="1000"/>
</p>


## 3.3 Correlations and STDP

Lorem ipsum dolor sit amet cum magna charta simplex ieronis
Lorem ipsum dolor sit amet cum magna charta simplex ieronis
Lorem ipsum dolor sit amet cum magna charta simplex ieronis
Lorem ipsum dolor sit amet cum magna charta simplex ieronis
Lorem ipsum dolor sit amet cum magna charta simplex ieronis
Lorem ipsum dolor sit amet cum magna charta simplex ieronis
Lorem ipsum dolor sit amet cum magna charta simplex ieronis


# Go back to:

[Chapter 1 : Leaky Integrate-and-Fire](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit1/README.md)

[Chapter 2 : Adaptations in Spiking Behavior](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit2/README.md)

[Chapter 3 : Spike-Timing Dependent Plasticity](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit3/README.md)

[Chapter 4: Synaptic Homeostasis](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit4/README.md)

[Chapter 4: Short-Term Plasticity](https://github.com/rtam97/C11-CompModNeuPlast/blob/main/results/unit5/README.md)

