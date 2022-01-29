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

        synaptic_args = ['E_exc' , 'w_exc', 'tau_e', 'E_inh', 'w_inh', 'tau_i']

        # Iterate through default values
        for key in synaptic_args:
            # Set synaptic attributes
            self.__set_synaptic_attributes(key)

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


            print('\n%s Excitatory synapses were added'%N)

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

            print('\n%s Inhibitory synapses were added'%N)

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

# ------------------------------- STIMULUS ----------------------------------- #

class Stimulus:

    def __init__(self, neuron, stim_type, **kwargs):

        # Parse parameters
        self.__parse_parameters(kwargs)

        # Fixed parameters from input
        self.type = stim_type
        self.neuron = neuron
        self.time = np.linspace(self.t_0, self.t_sim, int(self.t_sim / self.dt))
        self.stim_const = []


        # Initialize constant stimulus
        if self.type == StimulusType.CONSTANT.value:

            self.__generate_constant_stimulus()

        # Initialize periodic | poisson stimulus
        elif self.type == StimulusType.PERIODIC.value or self.type == StimulusType.POISSON.value:
            self.__create_stimuli(syntype='e')
            self.__create_stimuli(syntype='i')
            self.I_ext = 0.0
        else:
            sys.exit("Error: stimulus type must be either 'constant', 'periodic' or 'poisson' ")

    def __parse_parameters(self, arguments):

        defaultValues = dict(rate_exc = 6, rate_inh = 3,
                             t_0=0,t_sim=1000,dt=0.1,
                             stim_length=1,
                             I_ext=2.0,
                             stim_start=0,stim_end=1000)

        for key, defVal in defaultValues.items():
            setattr(self,key,arguments.get(key,defVal))

    def __create_stimuli(self,**kwargs):

        if self.type == StimulusType.CONSTANT.value:
            self.__generate_constant_stimulus()
        else:

            # Check synapse type
            if kwargs['syntype'] == 'e':
                N = self.neuron.N_exc
                rate = self.rate_exc
                name = ['RATE_EXC', 'N_EXC']
            elif kwargs['syntype'] == 'i':
                N = self.neuron.N_inh
                rate = self.rate_inh
                name = ['RATE_INH', 'N_INH']
            else:
                sys.exit('no buono')

            # Create stimulus
            stim = []
            if isinstance(rate, list):
                if len(rate) == N:
                    for n in range(N):
                        if self.type == StimulusType.PERIODIC.value:
                            stim.append(self.__generate_periodic_stimulus(rate[n]))
                        else:
                            stim.append([self.__generate_poisson_stimulus(rate[r], self.time, self.dt) for r in rate])
                elif len(rate) == 1:
                    if self.type == StimulusType.PERIODIC.value:
                        stim = [self.__generate_periodic_stimulus(rate[0])] * N
                    else:
                        stim = [self.__generate_poisson_stimulus(rate[0], self.time, self.dt) for s in range(N)]
                else:
                    sys.exit(f'Error: if {name[0]} is a LIST its length must be equal to {name[1]} or equal to 1')
            elif isinstance(rate, int):
                if self.type == StimulusType.PERIODIC.value:
                    stim = [self.__generate_periodic_stimulus(rate)] * N
                else:
                    stim = [self.__generate_poisson_stimulus(rate, self.time, self.dt) for s in range(N)]
            else:
                sys.exit(f"Error: {name[0]} must be either INT or LIST")

            # Assign stimulus to correct attribute
            if kwargs['syntype'] == 'e':
                self.stim_exc = stim
            elif kwargs['syntype'] == 'i':
                self.stim_inh = stim
            else:
                sys.exit('no buono')



    def __generate_constant_stimulus(self):

        if isinstance(self.I_ext,list):
            sys.exit('Constant current input can only take a single value')

        stim = np.zeros(len(self.time))
        stim[int(self.stim_start/self.dt):int(self.stim_end/self.dt)] = 1
        self.stim_const = stim*self.I_ext




    def __generate_periodic_stimulus(self, rate):
        time = np.linspace(self.t_0,self.t_sim,int(self.t_sim/self.dt))
        stimulus = np.zeros(len(time))

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
            stim_end_idx = [x + stim_length_idx for x in stim_start_idx]

            # Create range of indexes for stimulus duration
            for i, x in enumerate(stim_start_idx):
                stim_idx.append(list(range(stim_start_idx[i], stim_end_idx[i])))

            # Add inputs to stimulus vector
            try:
                stimulus[np.concatenate(stim_idx)] = 1
            except Exception as e:
                stimulus[-stim_length_idx:] = 1

        return stimulus

    def __generate_poisson_stimulus(self, rate, time, dt):
        stimulus = np.zeros(len(time))
        for t in range(len(time)):
            if rate * dt / 1000 > np.random.random():
                stimulus[t:t+int(1/dt)] = 1 # add a 1 ms stimulus
        return stimulus

class StimulusType(Enum):
    CONSTANT = 'constant'
    PERIODIC = 'periodic'
    POISSON = 'poisson'


# ------------------------------- SIMULATION ----------------------------------- #

class Simulation:
    def __init__(self,neuron,stimulus,**kwargs):

        self.neuron = neuron
        self.input = stimulus

        if 't_0' in kwargs.keys():
            self.t_0 = kwargs['t_0']
        elif hasattr(self.input,'t_0'):
            self.t_0 = self.input.t_0
        else:
            self.t_0 = 0

        if 't_sim' in kwargs.keys():
            self.t_sim = kwargs['t_sim']
        elif hasattr(self.input,'t_sim'):
            self.t_sim = self.input.t_sim
        else:
            self.t_sim = 1000

        if 'dt' in kwargs.keys():
            self.dt= kwargs['dt']
        elif hasattr(self.input,'dt'):
            self.dt = self.input.dt
        else:
            self.dt = 0.1

        self.trials = int(kwargs.get('trials', 1))

        if self.input.type != 'constant' and self.t_0 != self.input.t_0 and self.t_sim != self.input.t_sim and self.dt != self.input.dt:
            sys.exit('Error: the stimulus time and simulation time do not match!')

        # Time vector
        self.simtime = np.linspace(self.t_0, self.t_sim, int(self.t_sim / self.dt))

        # Empty voltage vector
        self.potential = np.zeros(len(self.simtime))

        # Empty excitatory conductance vector/matrix
        self.g_exc = np.zeros((neuron.N_exc,len(self.simtime)))

        # Empty inhibitory conductance vector/matrix
        self.g_inh = np.zeros((neuron.N_inh, len(self.simtime)))

        # Empty SRA conductance vector
        self.g_sra = np.zeros(len(self.simtime))

        # Empty Refractory Period conductance vector
        self.g_ref = np.zeros(len(self.simtime))

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


    def simulate(self,**kwargs):
        self.trials = int(kwargs.get('trials',1))

        if self.trials == 1:
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
        V[0] = self.neuron.V_init
        spikeCount = 0
        spikeTimes = []
        ISIs       = []
        sra_end     = 0
        spike = False

        # Integration loop
        for t in range(len(self.simtime) - 1):

            # Update EXCITATORY synaptic conductances
            if self.neuron.N_exc > 1:
                for m in range(self.neuron.N_exc):
                    g_ex[m][t + 1] = g_ex[m][t] + self.dt * self.neuron.synapse('e', g_ex[m][t], t, self.input, m)
            elif self.neuron.N_exc == 1:
                g_ex[0][t + 1] = g_ex[0][t] + self.dt * self.neuron.synapse('e', g_ex[0][t], t, self.input, 0)

            # Update EXCITATORY synaptic conductances
            if self.neuron.N_inh > 1:
                for n in range(self.neuron.N_inh):
                    g_in[n][t + 1] = g_in[n][t] + self.dt * self.neuron.synapse('i', g_in[n][t], t, self.input, n)
            elif self.neuron.N_inh == 1:
                g_in[0][t + 1] = g_in[0][t] + self.dt * self.neuron.synapse('i', g_in[0][t], t, self.input, 0)

            # Update POTENTIAL and spike/reset
            if V[t] <= self.neuron.V_theta:
                V[t + 1] = V[t] + self.dt * self.neuron.update_potential(V[t], self.input, time=t,
                                                                             g_exc=g_ex[:, t + 1], g_inh =g_in[:, t + 1],
                                                                             g_sra=g_sra[t], g_ref=g_ref[t])
            elif V[t] != self.neuron.V_spike:
                spike = True
                try:
                    sra_end = self.simtime[t+int(1/self.dt)]
                except Exception as e:
                    sra_end = self.simtime[-1]

                V[t + 1] = self.neuron.V_spike
                spikeCount += 1
                spikeTimes.append(self.simtime[t])
                if not ISIs:
                    ISIs.append(round(self.simtime[t],1))
                else:
                    ISIs.append(round(self.simtime[t]-spikeTimes[-2],1))
            else:
                V[t + 1] = self.neuron.V_reset

            if self.simtime[t] == sra_end:
                spike = False

            # Update SRA conductance and Refractory Period Conductance
            g_sra[t+1] = g_sra[t] + self.dt * self.neuron.adaptation('SRA',g_sra[t],spike)
            g_ref[t+1] = g_ref[t] + self.dt * self.neuron.adaptation('RP',g_ref[t], spike)


        # Save simulation results
        self.potential = V
        self.g_exc = g_ex
        self.g_inh = g_in
        self.g_sra = g_sra
        self.spikeCount = spikeCount
        self.spikeTimes = spikeTimes
        self.firingRate = spikeCount/self.t_sim*1000
        self.ISI       = ISIs
        self.outputFreq = (1/np.array(self.ISI))*1000 # Hz
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
                print('%s / %s '%(trial+1,trials))

            # Create new inputs with same parameters for each trials except 1st
            if trial != 0:
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