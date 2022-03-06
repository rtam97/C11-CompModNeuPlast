import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from time import time as sw
import sys
from enum import Enum


# ------------------------------- NEURON ----------------------------------- #

class Neuron:

    # Initialize neuron
    def __init__(self,type,**kwargs):

        # Neuron parameters
        self.type = type
        self.E_leak = kwargs.get('E_leak',-60)
        self.tau_m = kwargs.get('tau_m',20)
        self.R_m = kwargs.get('R_m', 10)

        # Spiking parameters
        self.V_init = kwargs.get('V_init', -70)
        self.V_reset = kwargs.get('V_reset', -70)
        self.V_theta = kwargs.get('V_theta', -50)
        self.V_spike = kwargs.get('V_spike', 0)

        # Excitatory synapse
        self.N_exc = kwargs.get('N_exc', 0)
        self.w_exc = kwargs.get('w_exc', 0.5)
        self.E_exc = kwargs.get('E_exc', 0)
        self.tau_e = kwargs.get('tau_e', 3)

        # Inhibitory synapse
        self.N_inh = kwargs.get('N_inh', 0)
        self.w_inh = kwargs.get('w_inh', 0.5)
        self.E_inh = kwargs.get('E_inh', -80)
        self.tau_i = kwargs.get('tau_i', 5)

        # Spike rate adaptation
        self.sra = kwargs.get('sra', 0)
        self.E_k = kwargs.get('E_k', -70)
        self.tau_s = kwargs.get('tau_s', 100)

        # Refractory period
        self.ref = kwargs.get('ref',0)
        self.E_rp = kwargs.get('E_rp',-70)
        self.tau_rp = kwargs.get('tau_rp',50)

        # Spike-Timing Dependend Plasticity EXCITATORY
        self.stdp = kwargs.get('stdp', self.stdp_types.NONE.value)
        self.tau_ltp_e = kwargs.get('tau_ltp_e', 17)
        self.tau_ltd_e = kwargs.get('tau_ltd_e', 34)

        if self.stdp == self.stdp_types.EXC.value or self.stdp == self.stdp_types.BOTH.value:
            self.A_ltp_e = kwargs.get('A_ltp_e',1.0)
            self.A_ltd_e = kwargs.get('A_ltd_e', -0.5)
        else:
            self.A_ltp_e = 0
            self.A_ltd_e = 0


        # Spike-Timing Dependend Plasticity INHIBITORY
        self.tau_ltp_i = kwargs.get('tau_ltp_i', 17)
        self.tau_ltd_i = kwargs.get('tau_ltd_i', 34)

        if self.stdp == self.stdp_types.INH.value or self.stdp == self.stdp_types.BOTH.value:
            self.A_ltp_i = kwargs.get('A_ltp_i', 1.0)
            self.A_ltd_i = kwargs.get('A_ltd_i', -0.5)
        else:
            self.A_ltp_i = 0
            self.A_ltd_i = 0

        # Synaptic normalization
        self.normalize  = kwargs.get('normalize',self.stdp_types.NONE.value)
        self.eta_norm   = kwargs.get('eta_norm',20)
        self.W_tot      = kwargs.get('W_tot',3)
        self.t_norm     = kwargs.get('t_norm', 10000) # 1 sec default

        # Intrinsic plasticity
        self.ip = kwargs.get('ip',self.stdp_types.NONE.value)
        self.eta_ip = kwargs.get('eta_ip',0)
        self.r_target = kwargs.get('r_target',3)


        # Short-Term Facilitation
        self.stf = kwargs.get('stf',self.stdp_types.NONE.value)
        self.U_stf = kwargs.get('U_stf',0.2)
        self.tau_stf = kwargs.get('tau_stf',50)
        self.w_fixed = kwargs.get('w_fixed',self.w_exc)
        self.w_affected = kwargs.get('w_affected',np.linspace(1,self.N_exc,self.N_exc)-1)

        synaptic_args = ['E_exc', 'w_exc', 'tau_e', 'A_ltp_e', 'A_ltd_e',
                         'E_inh', 'w_inh', 'tau_i', 'A_ltp_i', 'A_ltd_i']

        # Iterate through default values
        for key in synaptic_args:
            # Set synaptic attributes
            self.__set_synaptic_attributes(key)

        self.used_exc = 0
        self.used_inh = 0

        print('\nCreated Neuron')

    # Set attributes to synapse
    def __set_synaptic_attributes(self, key):

        # Excitatory attributes
        if key == 'E_exc':
            if not isinstance(self.E_exc, list):
                self.E_exc = [self.E_exc] * self.N_exc
        elif key == 'w_exc':
            if not isinstance(self.w_exc, list):
                self.w_exc = [self.w_exc] * self.N_exc
        elif key == 'tau_e':
            if not isinstance(self.tau_e, list):
                self.tau_e = [self.tau_e] * self.N_exc
        elif key == 'A_ltp_e':
            if not isinstance(self.A_ltp_e, list):
                self.A_ltp_e = [self.A_ltp_e] * self.N_exc
        elif key == 'A_ltd_e':
            if not isinstance(self.A_ltd_e, list):
                self.A_ltd_e = [self.A_ltd_e] * self.N_exc

        # Inhibitory attributes
        elif key == 'E_inh':
            if not isinstance(self.E_inh, list):
                self.E_inh = [self.E_inh] * self.N_inh
        elif key == 'w_inh':
            if not isinstance(self.w_inh, list):
                self.w_inh = [self.w_inh] * self.N_inh
        elif key == 'tau_i':
            if not isinstance(self.tau_i, list):
                self.tau_i = [self.tau_i] * self.N_inh
        elif key == 'A_ltp_i':
            if not isinstance(self.A_ltp_i, list):
                self.A_ltp_i = [self.A_ltp_i] * self.N_inh
        elif key == 'A_ltd_i':
            if not isinstance(self.A_ltd_i, list):
                self.A_ltd_i = [self.A_ltd_i] * self.N_inh

    # Add extra synapses to neuron
    def add_synapses(self,type,N,**kwargs):

        if type == 'E':

            # Add synapses
            self.N_exc += N


            # Resting potential (mV)
            if 'E_exc' in kwargs.keys():
                if not isinstance(kwargs['E_exc'], list):
                    self.E_exc = self.E_exc + [kwargs['E_exc']] * N
                elif len([kwargs['E_exc']] == N):
                    self.E_exc = self.E_exc + kwargs['E_exc']
                else:
                    self.E_exc = self.E_exc + [kwargs['E_exc']] * N
            else:
                self.E_exc = self.E_exc + [0] * N

            # Time constant (ms)
            if 'tau_e' in kwargs.keys():
                if not isinstance(kwargs['tau_e'], list):
                    self.tau_e = self.tau_e + [kwargs['tau_e']] * N
                elif len([kwargs['tau_e']] == N):
                    self.tau_e = self.tau_e + kwargs['tau_e']
                else:
                    self.tau_e = self.tau_e + [kwargs['tau_e']] * N
            else:
                self.tau_e = self.tau_e + [3] * N

            # Synaptic strength
            if 'w_exc' in kwargs.keys():
                if not isinstance(kwargs['w_exc'],list):
                    self.w_exc = self.w_exc + [kwargs['w_exc']] * N
                elif len([kwargs['w_exc']] == N):
                    self.w_exc = self.w_exc + kwargs['w_exc']
                else:
                    self.w_exc = self.w_exc + [kwargs['w_exc']] * N
            else:
                self.w_exc = self.w_exc + [0.5] * N


            # print('\n%s Excitatory synapses were added'%N)

        elif type == 'I':

            # Time constant (ms)
            if 'E_inh' in kwargs.keys():
                if not isinstance(kwargs['E_inh'], list):
                    self.E_inh = self.E_inh + [kwargs['E_inh']] * N
                elif len([kwargs['E_inh']] == N):
                    self.E_inh = self.E_inh + kwargs['E_inh']
                else:
                    self.E_inh = self.E_inh + [kwargs['E_inh']] * N
            else:
                self.E_inh = self.E_inh + [-80] * N

            # Time constant (ms)
            if 'tau_i' in kwargs.keys():
                if not isinstance(kwargs['tau_i'], list):
                    self.tau_i = self.tau_i + [kwargs['tau_i']] * N
                elif len([kwargs['tau_i']] == N):
                    self.tau_i = self.tau_i + kwargs['tau_i']
                else:
                    self.tau_i = self.tau_i + [kwargs['tau_i']] * N
            else:
                self.tau_i = self.tau_i + [5] * N

            # Synaptic strength
            if 'w_inh' in kwargs.keys():
                if not isinstance(kwargs['w_inh'], list):
                    self.w_inh = self.w_inh + [kwargs['w_inh']] * N
                elif len([kwargs['w_inh']] == N):
                    self.w_inh = self.w_exc + kwargs['w_inh']
                else:
                    self.w_inh = self.w_inh + [kwargs['w_inh']] * N
            else:
                self.w_inh = self.w_inh + [0.5] * N

            # print('\n%s Inhibitory synapses were added'%N)

        else:
            sys.exit('\nWARNING: \nSynapses must be either \n\tE -> Excitatory\n\tI -> Inhibitory')

    # Differential equation for membrane potential
    def update_potential(self, V, input, **kwargs):

        # External current
        if len(input.stim_const) > 0:
            I_ext = self.R_m*input.stim_const[kwargs['time']]
        else:
            I_ext = self.R_m*input.I_ext

        # Excitatory synaptic current
        if 'g_exc' in kwargs.keys():
            g_exc = kwargs['g_exc']
            I = g_exc*(np.array(self.E_exc) - V)
            I_exc = np.sum(I)
        else:
            I_exc = 0

        # Excitatory synaptic current
        if 'g_inh' in kwargs.keys():
            g_inh = kwargs['g_inh']
            I = g_inh * (np.array(self.E_inh) - V)
            I_inh = np.sum(I)
        else:
            I_inh = 0

        # Spike-rate adaptation
        if 'g_sra' in kwargs.keys():
            g_sra = kwargs['g_sra']
            I_sra = g_sra * (self.E_k - V)
        else:
            I_sra = 0

        # Refractory period
        if 'g_ref' in kwargs.keys():
            g_ref = kwargs['g_ref']
            I_ref = g_ref * (self.E_rp - V)
        else:
            I_ref = 0

        # Voltage update
        return (self.E_leak - V + I_ext + I_exc + I_inh + I_sra + I_ref)/self.tau_m

    # Differential equation for synaptic conductance
    def synapse(self,sType,g,t,stim,n):
        tau = 0
        w = 0
        t_stim = 0
        if sType == 'e':
            tau = self.tau_e[n]
            w = self.w_exc[n]
            t_stim = stim.stim_exc[n]
        elif sType == 'i':
            tau = self.tau_i[n]
            w = self.w_inh[n]
            t_stim = stim.stim_inh[n]

        return -g / tau + w * t_stim[t]

    # Differential equation for spike-rate and refractory adaptations
    def adaptation(self,adapt_type,g,spike):
        tau = 1
        dg = 0
        if adapt_type == 'SRA':
            tau = self.tau_s
            dg = self.sra
        elif adapt_type == 'RP':
            tau = self.tau_rp
            dg = self.ref
        return -g/tau + dg*spike

    # Spike timing dependent plasticity
    def STDP(self,syn,pre,spike,post,stim,n):
        if syn == 'e':
            A_ltp = self.A_ltp_e[n]
            A_ltd = self.A_ltd_e[n]
        else:
            A_ltp = self.A_ltp_i[n]
            A_ltd = self.A_ltd_i[n]
        return A_ltp * pre * spike + A_ltd * post * stim

    # Spike traces for STDP
    def spike_trace(self,syn,trace_type,x,spike):

        if trace_type == 'pre':
            if syn == 'e':
                tau = self.tau_ltp_e
            else:
                tau = self.tau_ltp_i
        else:
            if syn == 'e':
                tau = self.tau_ltd_e
            else:
                tau = self.tau_ltd_i

        return  -x / tau + spike

    # Spike traces with METHOD IV from Morrison 2008 [NOT USED]
    def spike_trace_2(self,type,x,spike,antispike,**kwargs):
        if type == 'pre':
            tau = self.tau_ltp
        else:
            tau = self.tau_ltd
            t = kwargs.get('t',0)
            ants = 0
            for m in range(self.N_exc):
                ants += antispike.stim_exc[m][t]
            for n in range(self.N_inh):
                ants += antispike.stim_inh[m][t]
            antispike = ants

        return  -x / tau + (1-x)*spike - x*antispike

    def synaptic_normalization(self,syn,w):

        sumweights=0
        if syn == self.stdp_types.EXC.value:
            sumweights = np.sum(self.w_exc)
        elif syn == self.stdp_types.INH.value:
            sumweights = np.sum(self.w_inh)

        return w*(1+self.eta_norm*(self.W_tot/sumweights - 1))

    # Intrinsic plasticity
    def adjust_threshold(self,R):
        return self.V_theta + self.eta_ip * (R - self.r_target)

    # STF
    def short_term_facilitation(self,u,spike):
        return -u/self.tau_stf + self.U_stf * (1-u) * spike

    # Print parameters of neuron and synapses
    def print_neuron_info(self):

        print(f"""
# ----------- NEURON PARAMETERS ----------- #')
Neuron Type                      :  {self.type}
Leak potential       (E_leak)    :  {self.E_leak}    mV  
Membrane resistance  (Rm)        :  {self.R_m}       MOhm 
Initial potential    (V_init)    :  {self.V_init}    mV 
Reset potential      (V_reset)   :  {self.V_reset}   mV 
Threshold potential  (V_theta)   :  {self.V_theta}   mV 
Spike potential      (V_spike)   :  {self.V_spike}   mV 

# ----------- SYNAPSE PARAMETERS ----------- #')
Excitatory synapses  (N_exc) :  {self.N_exc}
Reversal potential   (E_exc) :  {self.E_exc} mV
Time constant        (tau_e) :  {self.tau_e} ms
Synaptic strength    (w_exc) :  {self.w_exc}

Inhibitory synapses  (N_inh) :  {self.N_inh}
Reversal potential   (E_inh) :  {self.E_inh} mV
Time constant        (tau_i) :  {self.tau_i} ms
Synaptic strength    (w_inh) :  {self.w_inh}
                """)

    class stdp_types(Enum):
        EXC = 'e'
        INH = 'i'
        BOTH = 'both'
        NONE = 'none'

# ------------------------------- STIMULUS ----------------------------------- #

class Stimulus:

    def __init__(self, neuron, stim_type, **kwargs):

        # Fixed parameters from input
        self.type = stim_type
        self.neuron = neuron


        # Parse KWARGS
        self.S_e = kwargs.get('S_e', self.neuron.N_exc)     # Number of excitatory synapses to contact  (PERIODIC & POISSON)
        self.S_i = kwargs.get('S_i', self.neuron.N_inh)     # Number of inhibitory synapses to contact  (PERIODIC & POISSON)
        self.rate_exc   = kwargs.get('rate_exc',6)          # Rate of excitatory stimulus               (PERIODIC & POISSON)
        self.rate_inh   = kwargs.get('rate_inh',3)          # Rate of ihibitory stimulus                (PERIODIC & POISSON)
        self.stim_length = kwargs.get('stim_length',1)      # Length of each spike (ms)                 (PERIODIC & POISSON)
        self.corr_type = kwargs.get('corr_type', 'none')    # Correlation type                          (POISSON)
        self.corr = kwargs.get('corr', 0.0)                 # Amount of correlation between stimuli     (POISSON)
        self.tau_corr = kwargs.get('tau_corr', 20)          # Time constant for exp. correlated stimuli (POISSON)
        self.I_ext      = kwargs.get('I_ext',2.0)           # Amount of current injection               (CONSTANT)
        self.stim_start = kwargs.get('stim_start',0)        # Start time of current injection           (CONSTANT)
        self.stim_end   = kwargs.get('stim_end',1000)       # End time of current injection             (CONSTANT)
        self.t_0        = kwargs.get('t_0',0)               # Start time of simulation
        self.t_sim      = kwargs.get('t_sim',1000)          # End time of simulation
        self.dt         = kwargs.get('dt',0.1)              # Time step of simulation
        self.time = np.linspace(self.t_0, self.t_sim, int(self.t_sim / self.dt))

        if self.corr_type != self.CorrType.NONE.value and \
                self.corr_type != self.CorrType.INSTANTANEOUS.value and \
                self.corr_type != self.CorrType.EXPONENTIAL.value:
            sys.exit("Error: Correlation types must be one of: 'none', 'inst', 'exp'")



        # Empty lists to be filled
        self.stim_const = []
        self.stim_exc = []
        self.stim_inh = []
        self.stim_exc_times = []
        self.stim_inh_times = []

        # Generate input stimuli
        self.__create_stimuli()

    def __create_stimuli(self):
        print('\nGenerating stimulus ....')
        if self.type == self.StimulusType.CONSTANT.value:

            self.__generate_constant_stimulus()

        elif self.type == self.StimulusType.PERIODIC.value:

            self.I_ext = 0.0
            self.__synaptic_inputs(self.__generate_periodic_stimulus)

        elif self.type == self.StimulusType.POISSON.value:

            self.I_ext = 0.0
            self.__synaptic_inputs(self.__generate_poisson_stimulus)

        else:
            sys.exit("Error: stimulus type must be one of: 'constant', 'sinusoidal', 'periodic', 'poisson' ")

    def __synaptic_inputs(self, stimgen):

        for syntype in ['e', 'i']:
            # Check synapse type
            if syntype == 'e':
                N = self.neuron.N_exc
                to_add = self.S_e
                rate = self.rate_exc
                used = self.neuron.used_exc
                name = ['RATE_EXC', 'N_EXC']
            elif syntype == 'i':
                N = self.neuron.N_inh
                to_add = self.S_i
                rate = self.rate_inh
                used = self.neuron.used_inh
                name = ['RATE_INH', 'N_INH']
            else:
                sys.exit('Must define type of synapses to create inputs for')

            # Create the stimulus for each synapse
            if to_add > 0 and N > used and N >= used + to_add:
                stim, stimtimes = self.__make_stimulus(stimgen, rate, to_add)

                # Assign stimulus to correct attribute
                if syntype == 'e':
                    self.stim_exc = stim
                    self.stim_exc_times = stimtimes
                    self.neuron.used_exc += to_add
                elif syntype == 'i':
                    self.stim_inh = stim
                    self.stim_inh_times = stimtimes
                    self.neuron.used_inh += to_add
                else:
                    sys.exit('GOTTA BE E OR I')

    def __make_stimulus(self, func, rate, N):

        stim = []
        stimtimes = []
        stimindices = []

        # If rate is a list
        if isinstance(rate, list):
            if len(rate) == N:
                rate = rate
            elif len(rate) == 1:
                rate = rate * N
        elif isinstance(rate, int):
            rate = [rate] * N
        else:
            sys.exit('eee')

        # IF UNCORRELATED
        if self.corr_type == self.CorrType.NONE.value:
            stimulus = [func(rate[n]) for n in range(N)]
            # stimindices = [stimulus[i][2] for i in range(N)]
            stimtimes   = [stimulus[i][1] for i in range(N)]
            stim        = [stimulus[i][0] for i in range(N)]
        elif self.corr_type == self.CorrType.INSTANTANEOUS.value:
            stimulus = self.__correlate_poisson(rate,N)
            # stimindices = [stimulus[2][i] for i in range(N)]
            stimtimes   = [stimulus[1][i] for i in range(N)]
            stim        = [stimulus[0][i] for i in range(N)]
        elif self.corr_type == self.CorrType.EXPONENTIAL.value:
            stimulus = self.__correlate_poisson(rate,N)
            # stimindices = [stimulus[2][i] for i in range(N)]
            stimtimes   = [stimulus[1][i] for i in range(N)]
            stim        = [stimulus[0][i] for i in range(N)]

        return stim,stimtimes # ,stimindices

    def __generate_constant_stimulus(self):

        if isinstance(self.I_ext,list):
            sys.exit('Constant current input can only take a single value')

        stim = np.zeros(len(self.time))
        stim[int(self.stim_start/self.dt):int(self.stim_end/self.dt)] = 1
        self.stim_const = stim*self.I_ext

    def __generate_periodic_stimulus(self, rate):
        time = np.linspace(self.t_0,self.t_sim,int(self.t_sim/self.dt))
        stimulus = np.zeros(len(time))
        stimtimes = []
        stim_length_idx = int(self.stim_length/self.dt)
        stim_idx = []

        if rate != 0:

            # Inter spike intervals (constant)
            ISI = 1000/rate

            # Find time indexes of stimulus-start-points
                # This generates multiple points around the desired stimulus times, because of rounding
                # which causes inconsitencies in the length of the different stimuli
            stim_start_idx = [t for t in range(len(time) - 1)
                              if round(np.mod(round(time[t], 1), round(ISI, 1))) == 0.0
                              and round(time[t]) != 0.0]

            # Reject stimulus-start-points if they fall within the previous stimulus-length
                # fixes previous issue
            stim_start_idx = [x for i, x in enumerate(stim_start_idx)
                              if x - stim_start_idx[i - 1] > stim_length_idx
                              and i != 0]

            #
            # Find time indexes of stimulus end (depends on stim_length)
            stim_end_idx = [x + stim_length_idx+1 for x in stim_start_idx]

            # Create range of indexes for stimulus duration
            for i, x in enumerate(stim_start_idx):
                stim_idx.append(list(range(stim_start_idx[i], stim_end_idx[i])))

            # Add inputs to stimulus vector
            try:
                stimulus[np.concatenate(stim_idx)] = 1
                stimtimes = time[np.concatenate(stim_idx)]
            except Exception as e:
                stimulus[-stim_length_idx:] = 1
                stimtimes = time[-stim_length_idx:]

        return stimulus, stimtimes

    def __generate_poisson_stimulus(self, rate):
        stimulus = np.zeros(len(self.time))
        stimtimes = []
        stimidx = []
        for t in range(len(self.time)):
            if rate * self.dt / 1000 > np.random.random() and stimulus[t] != 1:
                stimulus[t:t+int(1/self.dt)+1] = 1 # add a 1 ms stimulus
                stimtimes.append(self.time[t])
                stimidx.append(t)
        return stimulus, np.array(stimtimes), np.array(stimidx)

    def __jitter_spike(self,spike_idx):
        shift = int(np.random.exponential(self.tau_corr)/self.dt)
        if spike_idx + shift > len(self.time):
            shift = 0
        sign = np.random.randint(-1,1)
        return spike_idx + shift

    def __correlate_poisson_old(self,rate,N):

        stim = []
        stimtimes = []
        stimindices = []

        # Generate source train with given rate
        rate_source = rate[0]
        source_train, source_times, source_idx = self.__generate_poisson_stimulus(rate_source)
        stim.append(source_train)
        stimtimes.append(source_times)
        stimindices.append(source_idx)

        for n in range(N-1):

            # Pick spike times/indices from source train with probability p
            p = np.sqrt(self.corr)
            new_idx = np.array([s for s in source_idx if np.random.uniform() < p])

            new_train = np.zeros(len(self.time))
            if self.corr_type == self.CorrType.EXPONENTIAL.value:
                new_idx = [self.__jitter_spike(s) for s in new_idx]


            new_train[new_idx]=1

            # Generate noise train to compensate for thinning rate loss
            rate_noise = rate_source*(1-p)
            noise_train, noise_times, noise_idx = self.__generate_poisson_stimulus(rate_noise)

            # Combine new indices and noise indices
            final_idx = np.sort(np.concatenate([new_idx,noise_idx]))

            # Generate final spike train
            final_train = np.zeros(len(self.time))
            final_times = []
            stim_length_idx = int(self.stim_length / self.dt)
            for t in final_idx:
                final_times.append(self.time[t])
                try:
                    final_train[t:t+stim_length_idx] = 1
                except:
                    final_train[t:] = 1

            stim.append(final_train)
            stimtimes.append(final_times)
            stimindices.append(final_idx)

            # plt.subplot(4,1,1)
            # plt.plot(source_train)
            # plt.subplot(4, 1, 2)
            # plt.plot(new_train)
            # plt.subplot(4, 1, 3)
            # plt.plot(noise_train)
            # plt.subplot(4, 1, 4)
            # plt.plot(final_train)
            # plt.show()

        return stim, stimtimes, stimindices

    def __correlate_poisson(self,rate,N):

        stim = []
        stimtimes = []
        stimindices = []

        # Generate source train with given rate
        rate_source = rate[0]
        source_train, source_times, source_idx = self.__generate_poisson_stimulus(rate_source)
        stim.append(source_train)
        stimtimes.append(source_times)
        stimindices.append(source_idx)

        n = 0
        while n < N-1:

            # Pick spike times/indices from source train with probability p
            p = np.sqrt(self.corr)
            new_idx = np.array([s for s in source_idx if np.random.uniform() < p])

            new_train = np.zeros(len(self.time))
            if self.corr_type == self.CorrType.EXPONENTIAL.value:
                new_idx = [self.__jitter_spike(s) for s in new_idx]


            new_train[new_idx]=1

            # Generate noise train to compensate for thinning rate loss
            rate_noise = rate_source*(1-p)
            noise_train, noise_times, noise_idx = self.__generate_poisson_stimulus(rate_noise)

            # Combine new indices and noise indices
            final_idx = np.sort(np.concatenate([new_idx,noise_idx]))

            # Generate final spike train
            final_train = np.zeros(len(self.time))
            final_times = []
            stim_length_idx = int(self.stim_length / self.dt)
            for t in final_idx:
                final_times.append(self.time[t])
                try:
                    final_train[t:t+stim_length_idx] = 1
                except:
                    final_train[t:] = 1

            stim.append(final_train)
            stimtimes.append(final_times)
            stimindices.append(final_idx)


            source_train = final_train
            source_idx = final_idx
            source_times = final_times
            n += 1

            # plt.subplot(4,1,1)
            # plt.plot(source_train)
            # plt.subplot(4, 1, 2)
            # plt.plot(new_train)
            # plt.subplot(4, 1, 3)
            # plt.plot(noise_train)
            # plt.subplot(4, 1, 4)
            # plt.plot(final_train)
            # plt.show()

        return stim, stimtimes, stimindices

    def add_stimulus(self,stim_type,**kwargs):
        return 1

    class StimulusType(Enum):
        CONSTANT = 'constant'
        PERIODIC = 'periodic'
        POISSON = 'poisson'

    class CorrType(Enum):
        NONE = 'none'
        INSTANTANEOUS = 'inst'
        EXPONENTIAL = 'exp'


# ------------------------------- SIMULATION ----------------------------------- #

class Simulation:

    def __init__(self,neuron,stimulus,**kwargs):

        self.neuron = neuron
        self.input = stimulus

        # Multiple stimuli
        if isinstance(self.input,list) and len(self.input) > 1:

            new_input = self.input[0]
            tmp_input_e = []
            tmp_times_e = []

            tmp_input_i = []
            tmp_times_i = []

            self.stim_group_boundaries = len(self.input[0].stim_exc) - 1

            for i in range(len(self.input)):
                tmp_input_e = tmp_input_e + self.input[i].stim_exc
                tmp_times_e = tmp_times_e + self.input[i].stim_exc_times

                tmp_input_i = tmp_input_i + self.input[i].stim_inh
                tmp_times_i = tmp_times_i + self.input[i].stim_inh_times

            new_input.stim_exc = tmp_input_e
            new_input.stim_exc_times = tmp_times_e

            new_input.stim_inh = tmp_input_i
            new_input.stim_inh_times = tmp_times_i

            if self.input[0].t_0 == self.input[1].t_0 and self.input[0].t_sim == self.input[1].t_sim and self.input[0].dt == self.input[1].dt:
                new_input.t_0 = self.input[0].t_0
                new_input.dt = self.input[0].dt
                new_input.t_sim = self.input[0].t_sim
            else:
                sys.exit('The time parameters must be equal for both stimuli')

            self.input = new_input

        self.t_0 = kwargs.get('t_0',(hasattr(self.input,'t_0'))*self.input.t_0)
        self.t_sim = kwargs.get('t_sim', (1-hasattr(self.input, 't_sim'))*1000 + # if nothing is given
                                (hasattr(self.input, 't_sim')) * self.input.t_sim) # if stimulus has time
        self.dt = kwargs.get('dt', (1 - hasattr(self.input, 'dt')) * 0.1 +  # if nothing is given
                                (hasattr(self.input, 'dt')) * self.input.dt)  # if stimulus has time

        self.trials = int(kwargs.get('trials', 1))

        if self.input.type != 'constant' and self.t_0 != self.input.t_0 and self.t_sim != self.input.t_sim and self.dt != self.input.dt:
            sys.exit('Error: the stimulus time and simulation time do not match!')

        # Time vector
        self.simtime = np.linspace(self.t_0, self.t_sim, int(self.t_sim / self.dt))

        # Empty voltage vector
        self.potential = np.zeros(len(self.simtime))

        # Empty excitatory conductance vector/matrix
        self.g_exc = np.zeros((neuron.N_exc, len(self.simtime)))

        # Empty inhibitory conductance vector/matrix
        self.g_inh = np.zeros((neuron.N_inh, len(self.simtime)))

        # Empty SRA conductance vector
        self.g_sra = np.zeros(len(self.simtime))

        # Empty Refractory Period conductance vector
        self.g_ref = np.zeros(len(self.simtime))

        # Empty Spike Timeing Trace vectors
        self.pre_trace_e = np.zeros((self.neuron.N_exc,len(self.simtime)))
        self.pre_trace_i = np.zeros((self.neuron.N_inh, len(self.simtime)))
        self.post_trace = np.zeros(len(self.simtime))

        # Vector of weights
        self.weights_e = np.array([np.full(len(self.simtime),self.neuron.w_exc[m]) for m in range(self.neuron.N_exc)])
        self.weights_i = np.array([np.full(len(self.simtime),self.neuron.w_inh[n]) for n in range(self.neuron.N_inh)])

        # Short term plasticity vectors
        self.stf_u_e = np.zeros((self.neuron.N_exc,len(self.simtime)))
        self.stf_u_i = np.zeros((self.neuron.N_inh,len(self.simtime)))

        # Empty counters
        self.spikeCount = 0
        self.spikeTimes = []
        self.firingRate = 0
        self.ISI       = []
        self.CV         = []
        self.meanISI    = 0
        self.meanCV     = 0
        self.meanSpikes = 0
        self.outputFreq = []
        self.fr_sec = []

    def simulate(self,**kwargs):
        self.trials = int(kwargs.get('trials',1))

        print('\nSimulating ...')

        if self.trials == 1:

            print('trial 1 / 1')
            self.__runSim()

        elif self.trials > 1:

            self.__runTrials(self.trials)

        else:
            sys.exit('You cannot simulate 0 or -Int trials')


    def __runSim(self):
        """
        Run a single trial simulation
        :return:
        """

        # Define initial vectors
        V = self.potential
        g_ex = self.g_exc
        g_in = self.g_inh
        g_sra = self.g_sra
        g_ref = self.g_ref
        pre_trace_e = self.pre_trace_e
        pre_trace_i = self.pre_trace_i
        post_trace = self.post_trace
        w_e = self.weights_e
        w_i = self.weights_i

        self.weights_sum = np.zeros(len(self.simtime))
        self.normfact = np.zeros(len(self.simtime))

        # Counters
        spikeCount = 0
        scIst = 0
        spikeTimes = []
        ISIs = []
        sra_end = 0
        spike = False
        # fr = []
        fr = np.zeros(len(self.simtime))
        theta = np.zeros(len(self.simtime))

        # initial value
        V[0] = self.neuron.V_init
        fr[0] = 0
        theta[0] = self.neuron.V_theta

        # Integration loop
        for t in range(len(self.simtime) - 1):

            # Print elapsed simulation time
            if np.mod(round(self.simtime[t],1),1000) == 0 and self.trials == 1:
                print(f'\r{round(self.simtime[t],1)/1000+1} / {self.t_sim/1000} s',end='')



            # Post-synaptic spike trace for STDP [== y(t) in Morrison et al, 2008]
            if self.neuron.stdp == 'both' or self.neuron.stdp == 'e' or self.neuron.stdp == 'i':
                post_trace[t + 1] = post_trace[t] + self.dt * self.neuron.spike_trace(syn='both', trace_type='post',
                                                                                      x = post_trace[t], spike = spike)
            elif self.neuron.stdp == 'none':
                pass
            else:
                sys.exit('Wrong STDP choice')

            if self.neuron.normalize != self.neuron.stdp_types.NONE.value:
                # print(np.sum([w[t] for w in w_e]))
                self.weights_sum[t] = (np.sum([w[t] for w in w_e]))
                self.normfact[t] = (1 + self.neuron.eta_norm*(self.neuron.W_tot/self.weights_sum[t] -1))

            # Update EXCITATORY synaspses
            for m in range(self.neuron.N_exc):


                # STDP
                if self.neuron.stdp == self.neuron.stdp_types.BOTH.value or self.neuron.stdp == self.neuron.stdp_types.EXC.value:

                    # SPIKE TRACE for pre-synaptic spikes [== x(t) in Morrison et al 2008]
                    pre_trace_e[m][t + 1] = pre_trace_e[m][t] + self.dt * self.neuron.spike_trace(syn='e', trace_type= 'pre',
                                                                                                  x = pre_trace_e[m][t],
                                                                                                  spike = self.input.stim_exc[m][t])
                    # STDP
                    if 6 > w_e[m][t] > 0:
                        w_e[m][t + 1] = w_e[m][t] + self.dt * self.neuron.STDP('e',pre_trace_e[m][t + 1], spike,
                                                                               post_trace[t + 1], self.input.stim_exc[m][t],m)
                    else:
                        w_e[m][t + 1] = w_e[m][t]

                # Synaptic Normalization
                if np.mod(round(self.simtime[t], 1), self.neuron.t_norm) == 0 and t != 0:
                    if (self.neuron.normalize == self.neuron.stdp_types.EXC.value
                            or self.neuron.normalize == self.neuron.stdp_types.BOTH.value):
                        w_e[m][t+1] = self.neuron.synaptic_normalization('e', w_e[m][t])


                # STF
                if (self.neuron.stf == self.neuron.stdp_types.BOTH.value or self.neuron.stf == self.neuron.stdp_types.EXC.value) \
                        and m in self.neuron.w_affected:
                    self.stf_u_e[m][t+1] = self.stf_u_e[m][t] + self.dt * self.neuron.short_term_facilitation(u=self.stf_u_e[m][t],spike=self.input.stim_exc[m][t])
                    if self.stf_u_e[m][t+1] != 0 and self.input.stim_exc[m][t]:
                        w_e[m][t + 1] = self.neuron.w_fixed*self.stf_u_e[m][t+1]
                    else:
                        w_e[m][t + 1] = w_e[m][t]

                # if round(self.simtime[t], 1) >= 1499.8 and m == 0:
                #     print(self.simtime[t])
                #     print(w_e[m][t], w_e[m][t + 1])

                self.neuron.w_exc[m] = w_e[m][t + 1]

                # CONDUCTANCE
                g_ex[m][t + 1] = g_ex[m][t] + self.dt * self.neuron.synapse('e', g_ex[m][t], t, self.input, m)


            # Update INHIBITORY synaptic conductances and STDP
            for m in range(self.neuron.N_inh):

                # STDP
                if self.neuron.stdp == self.neuron.stdp_types.BOTH.value or self.neuron.stdp == self.neuron.stdp_types.INH.value:
                    # SPIKE TRACE for pre-synaptic spikes =[x(t) in Morrison et al 2008]
                    pre_trace_i[m][t + 1] = pre_trace_i[m][t] + self.dt * self.neuron.spike_trace('i','pre', pre_trace_i[m][t],
                                                                                                  self.input.stim_inh[m][t])

                    if 6 > w_i[m][t]:
                        w_i[m][t + 1] = w_i[m][t] + self.dt * self.neuron.STDP('i',pre_trace_i[m][t + 1], spike,
                                                                               post_trace[t + 1], self.input.stim_inh[m][t],m)
                    else:
                        w_e[m][t + 1] = w_e[m][t]

                # Synaptic Normalization
                if np.mod(round(self.simtime[t], 1), 1000) == 0 and t != 0:
                    if (self.neuron.normalize == self.neuron.stdp_types.INH.value
                            or self.neuron.normalize == self.neuron.stdp_types.BOTH.value):
                        w_i[m][t + 1] = self.neuron.synaptic_normalization('i', w_i[m][t])

                # Update weight
                self.neuron.w_inh[m] = w_e[m][t + 1]

                # CONDUCTANCE
                g_in[m][t + 1] = g_in[m][t] + self.dt * self.neuron.synapse('i', g_in[m][t], t, self.input, m)


            # Update POTENTIAL and spike/reset
            if V[t] <= self.neuron.V_theta:

                # Update potential with LIF equation
                V[t + 1] = V[t] + self.dt * self.neuron.update_potential(V[t], self.input,
                                                                         time=t,
                                                                         g_exc=g_ex[:, t + 1],
                                                                         g_inh =g_in[:, t + 1],
                                                                         g_sra=g_sra[t],
                                                                         g_ref=g_ref[t])
            elif V[t] != self.neuron.V_spike:

                # has a spike occurred ? --> update counters/statistics
                spike = True
                spikeCount += 1
                scIst += 1
                spikeTimes.append(self.simtime[t])
                if not ISIs:
                    ISIs.append(round(self.simtime[t],1))
                else:
                    ISIs.append(round(self.simtime[t]-spikeTimes[-2],1))

                # Update potential to spiking potential
                V[t + 1] = self.neuron.V_spike

                # save time of when the SRA effect of current spike ends (t + 1 ms)
                try:
                    sra_end = self.simtime[t+int(1/self.dt)]
                except Exception as e:
                    sra_end = self.simtime[-1]
            else:
                # Reset potential
                V[t + 1] = self.neuron.V_reset

            # If the effect of SRA from previous spike is finished
            if self.simtime[t] == sra_end:
                spike = False

            # Update SRA conductance and Refractory Period Conductance
            if self.neuron.sra != 0 :
                g_sra[t+1] = g_sra[t] + self.dt * self.neuron.adaptation('SRA',g_sra[t],spike)
            if self.neuron.ref != 0 :
                g_ref[t+1] = g_ref[t] + self.dt * self.neuron.adaptation('RP',g_ref[t], spike)

            # Istantaneous firing rate
            if t != 0:
                fr[t+1] = spikeCount / self.simtime[t] * 1000

            # Second-by-secomd firing rate
            if np.mod(round(self.simtime[t],1),1000) == 0 and t != 0:
                self.fr_sec.append(scIst)
                scIst = 0
            elif t  == 0:
                self.fr_sec.append(0)
            elif t == len(self.simtime) - 2:
                self.fr_sec.append(scIst)




            if np.mod(round(self.simtime[t],1),1000) == 0 and self.neuron.eta_norm != 'none':
                # self.neuron.V_theta = self.neuron.adjust_threshold(fr[-1])
                self.neuron.V_theta = self.neuron.adjust_threshold(fr[t])
            theta[t+1] = self.neuron.V_theta



        # Save simulation results
        self.potential = V
        self.g_exc = g_ex
        self.g_inh = g_in
        self.g_sra = g_sra
        self.weights_e = w_e
        self.weights_i = w_i
        self.weights_sum[-1] = self.weights_sum[-2]
        self.fr = fr
        self.spikeCount = spikeCount
        self.spikeTimes = spikeTimes
        self.firingRate = spikeCount/self.t_sim*1000
        self.ISI       = ISIs
        self.outputFreq = (1/np.array(self.ISI))*1000 # Hz
        self.theta  = theta
        if spikeTimes:
            self.outputFreq[0] = 1/(spikeTimes[0]-self.simtime[int(self.input.stim_start/self.dt)])*1000
        if len(self.ISI) > 1:
            self.CV = np.std(self.ISI)/np.mean(self.ISI)
        else:
            self.CV = 0

    def __runTrials(self, trials):
        """
        Run a multi-trial simulation and compute statistics
        :return:
        """

        ISI = []
        CV = []
        SC = []
        FR = []
        potentials = []

        for trial in range(trials):

            if np.mod(trial,1) == 0:
                print('\n%s / %s '%(trial+1,trials))

            # Create new inputs with same parameters for each trials except 1st
            if trial != 0:
                self.input.neuron.used_exc = 0
                self.input.neuron.used_inh = 0
                stim_args = vars(self.input)
                self.input = Stimulus(stim_type=self.input.type,**stim_args)

            # Run individual trial
            self.__runSim()

            # store ISI, CV and spike-count from trial
            ISI.append(self.ISI)
            CV.append(self.CV)
            SC.append(self.spikeCount)
            FR.append(self.firingRate)
            potentials.append(self.potential)

        self.potential = potentials
        self.ISI = np.concatenate(ISI)
        self.CV = np.array(CV)
        tmp = self.CV[~np.isnan(self.CV)]
        self.CV = tmp[tmp != 0.0]
        self.spikeCount = np.array(SC)
        self.firingRate = FR
        self.meanCV = np.mean(self.CV)
        self.meanISI = np.mean(self.ISI)
        self.meanSpikes = np.round(np.mean(self.spikeCount))
        self.meanFR     = np.round(np.mean(self.firingRate),1)

    def plotISIdist(self,**kwargs):
        show = kwargs.get('show',False)
        expfit = kwargs.get('expfit',True)

        beans = 20

        plt.hist(self.ISI, fill=0, label='ISI',bins=beans)

        if expfit:
            def exp(x, a, b, c):
                return a * np.exp(-b * x) + c
            binval = np.histogram(self.ISI, bins=beans)[0]
            xval = np.linspace(0, np.max(self.ISI), beans)
            popt, pcov = curve_fit(exp, xval, binval, [1.0, 0.0, 1.0])
            plt.plot(xval, exp(xval, *popt), 'r-', label='exp.fit')

        plt.legend(loc='upper right')
        plt.xlabel('ISI')
        plt.ylabel('Frequency')
        plt.title('Mean ISI: %s ms' % np.round(self.meanISI, 1), fontweight='bold')

        if show:
            plt.show()

    def plotCVdist(self):
        if self.trials > 1 :
            plt.hist(self.CV, fill=0,bins=5)
            plt.xlabel('Coefficient of Variation (CV)')
            plt.title('Mean CV: %s ' % np.round(self.meanCV, 3), fontweight='bold')
        else:
            Warning('Cannot plot CV out of a single trial')

    def plotFiringRate(self,**kwargs):
        # Plot firing rate
        plt.plot(self.simtime/1000, self.fr, color='red',lw=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (Hz)')
        plt.title('Firing rate',fontweight='bold')
        if 'show' in kwargs.keys():
            if kwargs['show']:
                plt.show()

    def plotVoltageTrace(self,**kwargs):
        fulltrace = kwargs.get('fulltrace',False)

        if 'label' in kwargs.keys():
            if not fulltrace:
                if len(self.potential) == len(self.simtime):
                    plt.plot(self.simtime/1000, self.potential, label=kwargs['label'])
                else:
                    plt.plot(self.simtime / 1000, self.potential[-1], label=kwargs['label'])
            else:
                plt.plot(np.linspace(self.t_0,self.t_sim*self.trials,int(self.t_sim/self.dt))/1000,
                         np.concatenate(self.potential),
                         label=kwargs['label'])
            plt.legend(loc='upper left', fontsize='small')
        else:
            if not fulltrace:
                if len(self.potential) == len(self.simtime):
                    plt.plot(self.simtime / 1000, self.potential)
                else:
                    plt.plot(self.simtime / 1000, self.potential[-1])
            else:
                plt.plot(np.linspace(self.t_0, self.t_sim * self.trials, int(self.t_sim* self.trials / self.dt))/1000,
                         np.concatenate(self.potential))

        plt.xlabel('Time (s)')
        plt.ylabel('Membrane potential (mV)')
        if 'show' in kwargs.keys():
            if kwargs['show']:
                plt.show()
        return plt

    def plotSynapticWeights(self,**kwargs):

        syn = kwargs.get('syn',self.neuron.stdp_types.BOTH.value)

        W = []
        color = []
        lab = []
        if syn == self.neuron.stdp_types.BOTH.value:
            W = [self.weights_e, self.weights_i]
            color = ['green', 'orange']
            lab = ['Excitatory', 'Inhibitory']
        elif syn == self.neuron.stdp_types.EXC.value:
            W = [self.weights_e]
            color = ['green']
            lab = ['Excitatory']
        elif syn == self.neuron.stdp_types.INH.value:
            W = [self.weights_i]
            color = ['orange']
            lab = ['Inhibitory']

        for i,weights in enumerate(W):
            if weights.any():
                # Plot synaptic weights
                for w in weights:
                    plt.plot(self.simtime / 1000, w, color=f'tab:{color[i]}')

                if 'avg' in kwargs.keys():
                    if kwargs['avg']:
                        plt.plot(self.simtime / 1000, np.mean(weights, 0), color=color[i], lw=5,label=lab[i])

                plt.xlabel('Time (s)')
                plt.ylabel('Synaptic Weights')
                # plt.legend()

        if 'show' in kwargs.keys():
            if kwargs['show']:
                plt.show()

        return plt

    def plotInputCurrent(self,**kwargs):
        plt.plot(self.simtime/1000,self.input.stim_const,lw=2,color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (nA)')
        if 'show' in kwargs.keys():
            if kwargs['show']:
                plt.show()
        return plt

    def print_sim_parameters(self):
        print(f"""
# ----------- NEURON PARAMETERS ----------- #')
N_type  =  {self.neuron.type}
E_leak  =  {self.neuron.E_leak}   # Leak potential        (mV)
Rm      =  {self.neuron.R_m}      # Membrane resistance   (MOhm) 
V_init  =  {self.neuron.V_init}   # Initial potential     (mV)
V_reset =  {self.neuron.V_reset}  # Reset potential       (mV) 
V_theta =  {self.neuron.V_theta}  # Threshold potential   (mV)
V_spike =  {self.neuron.V_spike}  # Spike potential       (mV) 

# ----------- SYNAPSE PARAMETERS ----------- #')
N_exc   =  {self.neuron.N_exc} # Excitatory synapses
E_exc   =  {self.neuron.E_exc} # Reversal potential (mV)
tau_e   =  {self.neuron.tau_e} # Time constant (ms)
w_exc   =  {self.neuron.w_exc} # Synaptic strength
N_inh   =  {self.neuron.N_inh} # Inhibitory synapses
E_inh   =  {self.neuron.E_inh} # Reversal potential (mV)
tau_i   =  {self.neuron.tau_i} # Time constant (ms)
w_inh   =  {self.neuron.w_inh} # Synaptic strength


# ----------- INPUT PARAMETERS ----------- #
I_ext   = {self.input.I_ext}    # Input current (nA) 
rate_e  = {self.input.rate_exc} # Excitatory rates 
rate_i  = {self.input.rate_inh} # Inhibitory rates  


# ----------- SIMULATION PARAMETERS ----------- #
t_0     = {self.t_0}    # Start simulation time (ms) 
t_sim   = {self.t_sim}  # Simulation duration (ms)
dt      = {self.dt}     # Integration time step (ms)
""")

    def print_sim_stats(self):
        print(f"""
# ----------- SIMULATION STATS ----------- #')
Number of spikes    =  {self.spikeCount}
Firing rate         =  {self.firingRate} Hz
Average ISI         =  {self.meanISI} ms 
CV                  =  {self.CV}  
""")

    def saveInputParameters(self,file):
        with open(file,'w') as f:
            f.write(f"""
# ----------- NEURON PARAMETERS ----------- #')
N_type  =  {self.neuron.type}
E_leak  =  {self.neuron.E_leak}   # Leak potential        (mV)
Rm      =  {self.neuron.R_m}      # Membrane resistance   (MOhm) 
V_init  =  {self.neuron.V_init}   # Initial potential     (mV)
V_reset =  {self.neuron.V_reset}  # Reset potential       (mV) 
V_theta =  {self.neuron.V_theta}  # Threshold potential   (mV)
V_spike =  {self.neuron.V_spike}  # Spike potential       (mV) 

# ----------- SYNAPSE PARAMETERS ----------- #')
N_exc   =  {self.neuron.N_exc} # Excitatory synapses
E_exc   =  {self.neuron.E_exc} # Reversal potential (mV)
tau_e   =  {self.neuron.tau_e} # Time constant (ms)
w_exc   =  {self.neuron.w_exc} # Synaptic strength
N_inh   =  {self.neuron.N_inh} # Inhibitory synapses
E_inh   =  {self.neuron.E_inh} # Reversal potential (mV)
tau_i   =  {self.neuron.tau_i} # Time constant (ms)
w_inh   =  {self.neuron.w_inh} # Synaptic strength


# ----------- INPUT PARAMETERS ----------- #
I_ext   = {self.input.I_ext}    # Input current (nA) 
rate_e  = {self.input.rate_exc} # Excitatory rates 
rate_i  = {self.input.rate_inh} # Inhibitory rates  


# ----------- SIMULATION PARAMETERS ----------- #
t_0     = {self.t_0}    # Start simulation time (ms) 
t_sim   = {self.t_sim}  # Simulation duration (ms)
dt      = {self.dt}     # Integration time step (ms)
""")

    def save_checkpoint(self):
        sim = vars(self)

        stim = vars(self.input)

        neur = vars(self.neuron)

        sim['neuron'] = 'placeholder'

        stim['neuron'] = 'placeholder'

        sim['input'] = 'placeholder'

        with open('cpt.py','w') as f:
            f.write(f'neuron = {neur}\n\n')
            f.write(f'stim = {stim}\n\n')
            f.write(f'sim = {sim}\n\n')

    def restore_checkpoint(self,file):

        with open(file,'w') as f:
            print(1)

        print(2)


# ------------------------------- STATS ----------------------------------- #


def cross_correlogram(trains,bin_range,dt,type):

    bins = abs(bin_range[0]) + abs(bin_range[1]) + 1
    # bins = 100
    M = np.linspace(bin_range[0], bin_range[1], bins)
    Hm = []
    delay = []

    n = 0
    if type == 'within':

        for i in range(len(trains)):
            # print(f'Train {i+1} / {len(trains)}')
            for spike1 in trains[i]:
                for j in range(len(trains)):
                    if j != i:
                        for spike2 in trains[j]:
                            delay.append(spike1-spike2)
                            n += 1
    elif type == 'between':
        if len(trains) == 2:
            for i, train in enumerate(trains[0]):
                # print(f'Group 1 Train {i+1} / {len(trains[0])}')
                for spike1 in train:
                    for j, train2 in enumerate(trains[1]):
                        for spike2 in train2:
                            delay.append(spike1-spike2)
                            n += 1
        else:
            sys.exit('Error: to compare between two groups TRAINS must be a list of 2 elements, each of which is a list')
    else:
        sys.exit('Error: Either within or between groups analysis')

    for m in M:
        Nm = 0
        for d in delay:
            if (m - 1 / 2) * dt < d < (m + 1 / 2) * dt:
                Nm += 1
        Hm.append(Nm)#/n*100 )
    return M, Hm