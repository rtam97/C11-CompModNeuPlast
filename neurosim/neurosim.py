import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from time import time as sw
import sys


# ------------------------------- NEURON ----------------------------------- #

class Neuron:

    # Define neuron parameters and synaptic inputs
    def __init__(self,type,**kwargs):

        self.type = type

        # -------------------------------------------- NEURON PARAMETERS --------------------------------------------- #

        # Leak reversal potential  (mV)
        if 'E_leak' in kwargs.keys():
            self.E_leak  = kwargs['E_leak']
        else:
            self.E_leak = -60

        # Membrane time constant (ms)    --> Rm*Cm [implicit]
        if 'tau_m' in kwargs.keys():
            self.tau_m = kwargs['tau_m']
        else:
            self.tau_m = 20

        # Membrane resistance      (Mohm)
        if 'R_m' in kwargs.keys():
            self.R_m = kwargs['R_m']
        else:
            self.R_m = 10

        # Initial potential     (mV)
        if 'V_init' in kwargs.keys():
            self.V_init = kwargs['V_init']
        else:
            self.V_init = -70

        # Reset potential       (mV)
        if 'V_reset' in kwargs.keys():
            self.V_reset = kwargs['V_reset']
        else:
            self.V_reset = -70

        # Threshold potential   (mV)
        if 'V_theta' in kwargs.keys():
            self.V_theta = kwargs['V_theta']
        else:
            self.V_theta = -50

        # Spike amplitude       (mV)
        if 'V_spike' in kwargs.keys():
            self.V_spike = kwargs['V_spike']
        else:
            self.V_spike = 0

        # ------------------------------------------- EXCITATORY SYNAPSES -------------------------------------------- #


        # Excitatory synapses
        if 'N_exc' in kwargs.keys() and kwargs['N_exc'] != 0:

            self.N_exc  = kwargs['N_exc']

            # Reversal potential (mV)
            if 'E_exc' in kwargs.keys():                        # IF :  reversal potential is provided
                if not isinstance(kwargs['E_exc'],list):            # IF : only one value is given as int/float
                    self.E_exc = [kwargs['E_exc']]*self.N_exc
                elif len(kwargs['E_exc']) == self.N_exc:            # IF : nr of values == nr of synapses (N)
                    self.E_exc = kwargs['E_exc']
                else:                                               # ELSE : default value N times
                    self.E_exc = [0] * self.N_exc
            else:                                               # ELSE: default value N times
                self.E_exc = [0]* self.N_exc

            # Time constant (ms)
            if 'tau_e' in kwargs.keys():

                if not isinstance(kwargs['tau_e'],list):
                    self.tau_e = [kwargs['tau_e']]*self.N_exc
                elif len(kwargs['tau_e']) == self.N_exc:
                    self.tau_e = kwargs['tau_e']
                else:
                    self.tau_e = [3] * self.N_exc
            else:
                self.tau_e = [3] * self.N_exc

            # Synaptic strength
            if 'w_exc' in kwargs.keys():
                if not isinstance(kwargs['w_exc'],list):
                    self.w_exc = [kwargs['w_exc']]*self.N_exc
                elif len(kwargs['w_exc']) == self.N_exc:
                    self.w_exc = kwargs['w_exc']
                else :
                    self.w_exc = [0.5]*self.N_exc
            else:
                self.w_exc = [0.5]*self.N_exc
        else:
            self.N_exc = 0
            self.w_exc = []
            self.E_exc = []
            self.tau_e = []

        # ------------------------------------------- INHIBITORY SYNAPSES -------------------------------------------- #
        # Inhibitory synapses
        if 'N_inh' in kwargs.keys():

            self.N_inh = kwargs['N_inh']

            # Reversal potential (mV)
            if 'E_inh' in kwargs.keys():
                if not isinstance(kwargs['E_inh'],list):
                    self.E_inh = [kwargs['E_inh']] * self.N_inh
                elif len([kwargs['E_inh']]) == self.N_inh:
                    self.E_inh = kwargs['E_inh']
                else:
                    self.E_inh = [-80] * self.N_inh
            else:
                self.E_inh = [-80] * self.N_inh

            # Time constant (ms)
            if 'tau_i' in kwargs.keys():
                if not isinstance(kwargs['tau_i'], list):
                    self.tau_i = [kwargs['tau_i']] * self.N_inh
                elif len([kwargs['tau_i']]) == self.N_inh:
                    self.tau_i = kwargs['tau_i']
                else:
                    self.tau_i = [kwargs['tau_i']] * self.N_inh
            else:
                self.tau_i = [5] * self.N_inh

            # Synaptic strength
            if 'w_inh' in kwargs.keys():
                if not isinstance(kwargs['w_inh'], list):
                    self.w_inh = [kwargs['w_inh']]  * self.N_inh
                elif len(kwargs['w_inh']) == self.N_inh:
                    self.w_inh = [kwargs['w_inh']]
                else:
                    self.w_inh = [kwargs['w_inh']] * self.N_inh
            else:
                self.w_inh = [0.5] * self.N_inh
        else:
            self.N_inh = 0
            self.w_inh = []
            self.E_inh = []
            self.tau_i = []

    # Add synapses to neuron
    def addSynapses(self,type,N,**kwargs):

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
            print('\nWARNING: \nSynapses must be either \n\tE -> Excitatory\n\tI -> Inhibitory')

        return 0

    # Differential equation for membrane potential
    def update_potential(self, V, input, **kwargs):

        # External current
        I_ext = self.R_m*input.I_ext

        # Excitatory synaptic current
        if 'g_exc' in kwargs.keys():
            g_exc = kwargs['g_exc']
            I = g_exc*(np.array(self.E_exc) - V)
            I_exc = np.sum(I)

        else:
            I_ext = 0

        # Excitatory synaptic current
        if 'g_inh' in kwargs.keys():
            g_inh = kwargs['g_inh']
            I = g_inh * (np.array(self.E_inh) - V)
            I_inh = np.sum(I)
        else:
            I_inh = 0

        # Voltage update
        return (self.E_leak - V + I_ext + I_exc + I_inh)/self.tau_m

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

    # Print parameters of neuron and synapses
    def printNeuronInfo(self):
        print('\n# ----------- NEURON PARAMETERS ----------- #')
        print('Neuron Type                      : ', self.type)
        print('Leak potential       (E_leak)    : ', self.E_leak    ,   ' mV ' )
        print('Membrane resistance  (Rm)        : ', self.R_m       ,   ' MOhm ')
        print('Initial potential    (V_init)    : ', self.V_init    ,   ' mV ')
        print('Reset potential      (V_reset)   : ', self.V_reset   ,   ' mV ')
        print('Threshold potential  (V_theta)   : ', self.V_theta   ,   ' mV ')
        print('Spike potential      (V_spike)   : ', self.V_spike   ,   ' mV ')

        print('\n# ----------- SYNAPSE PARAMETERS ----------- #')
        print('Excitatory synapses  (N_exc) : ', self.N_exc)
        print('Reversal potential   (E_exc) : ', self.E_exc, ' mV ')
        print('Time constant        (tau_e) : ', self.tau_e, ' ms ')
        print('Synaptic strength    (w_exc) : ', self.w_exc)

        print('\nInhibitory synapses  (N_inh) : ', self.N_inh)
        print('Reversal potential   (E_inh) : ', self.E_inh, ' mV ')
        print('Time constant        (tau_i) : ', self.tau_i, ' ms ')
        print('Synaptic strength    (w_inh) : ', self.w_inh)

        return 0


# ------------------------------- INPUT ----------------------------------- #

class Stimulus:
    def __init__(self,neuron,type,**kwargs):

        self.type = type
        self.parse_parameters(kwargs)

        # -------------------------- CONSTANT STIMULUS -------------------------- #
        if type == 'constant':

            if 'I_ext' in kwargs.keys():
                if isinstance(kwargs['I_ext'],list):
                    print('Constant current input can only take a single value')
                else:
                    self.I_ext = kwargs['I_ext']
            else :
                self.I_ext = 2.0

        # -------------------------- PERIODIC STIMULUS -------------------------- #
        elif type == 'periodic':

            # ......................... EXCITATORY STIMULI ......................... #
            Ne = neuron.N_exc
            rate = self.rate_exc
            self.stim_exc = []
            if isinstance(rate,list):
                if len(rate) == Ne:
                    for n in range(Ne):
                        self.stim_exc.append(self.generate_periodic_stimulus(rate[n]))
                elif len(rate) == 1:
                    self.stim_exc = [self.generate_periodic_stimulus(rate[0])] * Ne
                else:
                    sys.exit('Error: if RATE_EXC is a LIST its length must be equal to N_EXC or equal to 1')
            elif isinstance(rate,int):
                self.stim_exc = [self.generate_periodic_stimulus(rate)] * Ne
            else:
                sys.exit("Error: RATE_EXC must be either INT or LIST")

            # ......................... INHIBITORY STIMULI ......................... #
            Ni = neuron.N_inh
            rate = self.rate_inh
            self.stim_inh = []
            if isinstance(rate, list):
                if len(rate) == Ni:
                    for n in range(Ni):
                        self.stim_inh.append(self.generate_periodic_stimulus(rate[n]))
                elif len(rate) == 1:
                    self.stim_inh = [self.generate_periodic_stimulus(rate[0])] * Ni
                else:
                    sys.exit('Error: if RATE_INH is a LIST its length must be equal to N_INH or 1')
            elif isinstance(rate, int):
                self.stim_inh = [self.generate_periodic_stimulus(rate)] * Ni
            else:
                sys.exit("Error: RATE_INH must be either INT or LIST")



        # -------------------------- POISSON STIMULUS -------------------------- #
        elif type == 'poisson':



            # ......................... EXCITATORY STIMULI ......................... #
            Ne = neuron.N_exc
            rate = self.rate_exc
            self.stim_exc = []
            time = np.linspace(self.t_0, self.t_sim, int(self.t_sim / self.dt))

            if isinstance(rate, list):
                if len(rate) == Ne:
                    for n in range(Ne):
                        self.stim_exc.append([self.generate_poisson_stimulus(rate[n], time, self.dt) for n in rate])
                elif len(rate) == 1:
                    self.stim_exc = [self.generate_poisson_stimulus(rate[0], time, self.dt) for s in range(Ne)]
                else:
                    sys.exit('Error: if RATE_EXC is a LIST its length must be equal to N_EXC or equal to 1')
            elif isinstance(rate, int):
                self.stim_exc = [self.generate_poisson_stimulus(rate, time, self.dt) for s in range(Ne)]
            else:
                sys.exit("Error: RATE_EXC must be either INT or LIST")

            # ......................... INHIBITORY STIMULI ......................... #
            Ni = neuron.N_inh
            rate = self.rate_inh
            self.stim_inh = []
            if isinstance(rate, list):
                if len(rate) == Ni:
                    for n in range(Ni):
                        self.stim_inh.append([self.generate_poisson_stimulus(rate[m], time, self.dt) for m in rate])
                elif len(rate) == 1:
                    self.stim_inh = [self.generate_poisson_stimulus(rate[0], time, self.dt) for s in range(Ni)]
                else:
                    sys.exit('Error: if RATE_INH is a LIST its length must be equal to N_INH or 1')
            elif isinstance(rate, int):
                self.stim_inh = [self.generate_poisson_stimulus(rate, time, self.dt) for s in range(Ni)]
            else:
                sys.exit("Error: RATE_INH must be either INT or LIST")




            self.I_ext = 0

    def parse_parameters(self,arguments):


        # Firing rate EXCITATORY (Hz)
        if 'rate_exc' in arguments.keys():
            self.rate_exc = arguments['rate_exc']
        else:
            self.rate_exc = 6

        # Firing rate INHIBITORY (Hz)
        if 'rate_inh' in arguments.keys():
            self.rate_inh = arguments['rate_inh']
        else:
            self.rate_inh = 3

        # Duration of each stimulus (ms)
        if 'stim_length' in arguments.keys():
            self.stim_length = arguments['stim_length']
        else:
            self.stim_length = 1

        # Start time
        if 't_0' in arguments.keys():
            self.t_0 = arguments['t_0']
        else:
            self.t_0 = 1
        self.t_0 = 0

        # Simulation time (ms)
        if 't_sim' in arguments.keys():
            self.t_sim = arguments['t_sim']
        else:
            self.t_sim = 1000

        # Time step (ms)
        if 'dt' in arguments.keys():
            self.dt = arguments['dt']
        else:
            self.dt = 0.1

        # Input current (nA)
        if 'I_ext' in arguments.keys():
            self.I_ext = arguments['I_ext']
        else:
            self.I_ext = 0

        return 0

    def generate_poisson_stimulus(self, rate, time, dt):
        stimulus = np.zeros(len(time))
        for t in range(len(time)):
            if rate * dt / 1000 > np.random.random():
                stimulus[t:t+int(1/dt)] = 1 # add a 1 ms stimulus
        return stimulus

    def generate_periodic_stimulus(self,rate):
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

    def getParams(self):
        return self.dt

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

        # Empty counters
        self.spikeCount = 0
        self.spikeTimes = []
        self.firingRate = 0
        self.ISI       = []
        self.CV         = []
        self.meanISI    = 0
        self.meanCV     = 0
        self.meanSpikes = 0


    def simulate(self,**kwargs):
        self.trials = int(kwargs.get('trials',1))

        if self.trials == 1:
            self.runSim()
        elif self.trials > 1:
            self.runTrials(self.trials)
        else:
            sys.exit('You cannot simulate 0 or -Int trials')


    def runSim(self):
        """
        Run a single trial simulation
        :return:
        """

        # Define voltage vector
        V = self.potential
        g_ex = self.g_exc
        g_in = self.g_inh
        V[0] = self.neuron.V_init
        spikeCount = 0
        spikeTimes = []
        ISIs       = []

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
                V[t + 1] = V[t] + self.dt * self.neuron.update_potential(V[t], self.input, g_exc=g_ex[:,t+1], g_inh = g_in[:,t+1])
            elif V[t] != self.neuron.V_spike:
                V[t + 1] = self.neuron.V_spike
                spikeCount += 1
                spikeTimes.append(self.simtime[t])
                if not ISIs:
                    ISIs.append(round(self.simtime[t],1))
                else:
                    ISIs.append(round(self.simtime[t]-spikeTimes[-2],1))
            else:
                V[t + 1] = self.neuron.V_reset

        # Save simulation old_results
        self.potential = V
        self.g_exc = g_ex
        self.g_inh = g_in
        self.spikeCount = spikeCount
        self.spikeTimes = spikeTimes
        self.firingRate = spikeCount/self.t_sim*1000
        self.ISI       = ISIs
        self.CV         = np.std(self.ISI)/np.mean(self.ISI)


    def runTrials(self,trials):
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
                self.input = Stimulus(neuron=self.neuron,**stim_args)

            # Run individual trial
            self.runSim()

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
            f.write('\n# ----------- NEURON PARAMETERS ----------- #')
            f.write('\nNeuron Type=%s'%self.neuron.type)
            f.write('\nE_leak    = %s # Leak reversal potential   (mV)'%self.neuron.E_leak)
            f.write('\nRm        = %s # Membrane resistance       (Mohm)'%self.neuron.R_m)
            f.write('\nV_init    = %s # Initial potential         (mV)'%self.neuron.V_init)
            f.write('\nV_reset   = %s # Reset potential           (mV)'%self.neuron.V_reset)
            f.write('\nV_theta   = %s # Threshold potential       (mV)'%self.neuron.V_theta)
            f.write('\nV_spike   = %s # Spiking potential         (mV)'%self.neuron.V_spike)

            f.write('\n\n# ----------- SYNAPSE PARAMETERS ----------- #')
            f.write('\nN_exc = %s # Inhibitory synapses'%self.neuron.N_exc)
            f.write('\nE_exc = %s # Reversal potential (mV)'%self.neuron.E_exc)
            f.write('\ntau_e = %s # Time constant (ms)'%self.neuron.tau_e)
            f.write('\nw_exc = %s # Synaptic strength '%self.neuron.w_exc)

            f.write('\n\nN_inh = %s # Inhibitory synapses'%self.neuron.N_inh)
            f.write('\nE_inh = %s # Reversal potential (mV)'%self.neuron.E_inh)
            f.write('\ntau_i = %s # Time constant (ms)'%self.neuron.tau_i)
            f.write('\nw_inh = %s # Synaptic strength'%self.neuron.w_inh)

            f.write('\n\n# ----------- INPUT PARAMETERS ----------- #')
            f.write('\nI_ext = %s # Input current (nA)' % self.input.I_ext)
            f.write('\nrate_exc = %s # Excitatory rates' % self.input.rate_exc)
            f.write('\nrate_inh = %s # Inhibitory rates' % self.input.rate_inh)


            f.write('\n\n# ----------- SIMULATION PARAMETERS ----------- #')
            f.write('\nt_0    = %s # Start simulation time (ms)' % self.t_0)
            f.write('\nt_sim  = %s # Simulation duration (ms)' % self.t_sim)
            f.write('\ndt     = %s # Integration time step (ms)'% self.dt)


