# %%

import numpy as np
import matplotlib as mlp
from matplotlib import pyplot as plt
from scripts.solvers import eulerNeuron as solve
from scripts.neurons import *
from time import time as stopwatch
from scipy.optimize import curve_fit
# from matplotlib import use
# use('TkAgg')

mlp.rcParams['figure.figsize'] = (8,8)
# --------------------------------- PARAMETERS ---------------------------------

from scripts.parameters.param14_poisson import *

ISI_total = []
CV_total = []

WEX = np.linspace(0.5,1.5,11)

for fig_idx,w_ex in enumerate(WEX):
    params = {
        'E_leak': E_leak,
        'tau_m': tau_m,

        'E_ex': E_ex,
        'w_ex': w_ex,
        'tau_e': tau_e,
        'f_ex' : f_ex, # Firing rate EXC
        't_ex' : t_ex,

        'E_in': E_in,
        'w_in': w_in,
        'tau_i': tau_i,
        'f_in' : f_in, # Firing rate INH
        't_in' : t_in
    }

    # --------------------------------- SIMULATIONS --------------------------------- #

    trials = 50

    pot = []
    ISI = []
    CV = []
    SC = []


    start = stopwatch()
    for trial in range(trials):
        trial_start = stopwatch()
        print('Trial %s / %s'%(trial+1,trials))
        (V, g_ex, g_in, time, spikeCount, spikeTimes) = solve(neuron=LIF_inputs,
                                                             synapse=g_synapse,
                                                             Ne=10,
                                                             Ni=10,
                                                             dt=dt,
                                                             t_end=t_end,
                                                             init=V_reset,
                                                             threshold=V_theta,
                                                             spiking=V_spike,
                                                             reset=V_reset,
                                                             stim_type='poisson',
                                                             **params)

        # Compute trial statistics
        if spikeCount > 1:
            ISI_trial = [spikeTimes[t] - spikeTimes[t-1] for t in range(1,len(spikeTimes))]
            ISI_trial.insert(0,spikeTimes[0])
            CV_trial = np.std(ISI_trial)/np.mean(ISI_trial)
            ISI.append(ISI_trial)
            CV.append(CV_trial)

        # Store trial data
        pot.append(V)
        SC.append(spikeCount)


        trial_end = stopwatch()
        print('\t%s'%(trial_end-trial_start))

    end = stopwatch()
    print(end-start)


    # --------------------------------- COMPUTE STATISTICS ---------------------------------


    # Combine trial voltage and time in a single time series
    V_combined = np.concatenate([v for v in pot])
    time_combined = np.linspace(0,t_end*trials,len(V_combined))

    # Compute statistics
    spikes = int(np.round(np.mean(SC)))
    ISI = np.squeeze(np.concatenate(ISI))
    CV = np.squeeze(CV)
    CV = CV[~np.isnan(CV)]
    print('spikes :',np.sum(SC))
    print('ISIs   :',len(ISI))
    print('CVs   :',len(CV))

    # Exponential fit
    beans = 15
    def exp(x,a,b,c):
        return a*np.exp(-b*x)+c
    binval = np.histogram(ISI, bins=beans)[0]
    xval = np.linspace(0,np.max(ISI),beans)
    popt,pcov = curve_fit(exp,xval,binval,[1, 0, 0])

    # --------------------------------- PLOT FIGURES --------------------------------- #

    # Plot combined voltage time series
    plt.subplot2grid((2,2),(0,0),colspan=2)
    # plt.plot(time_combined/1000,V_combined)
    plt.plot(time/1000,V)
    plt.xlabel('Time (s)')
    plt.ylabel('Memb. potential (mV)')
    title = 'We: %s || Wi: %s || Trials: %s x %s s || Rate: %s Hz || Mean #Spikes: %s'\
                %(w_ex,w_in,
                  trials, t_end/1000,
                  f_ex,
                  spikes)
    plt.title(title,fontweight='bold')

    # Plot ISI histogram
    plt.subplot2grid((2, 2), (1, 0), colspan=1)
    plt.hist(ISI, fill=0, bins=beans, label='ISI')
    plt.plot(xval, exp(xval, *popt), 'r-', label='exp.fit')
    plt.legend(loc='upper right')
    plt.xlabel('ISI')
    plt.ylabel('Frequency')
    plt.title('Mean ISI: %s ms '%(np.round(np.mean(ISI),3)),fontweight='bold')

    # Plot CVs histogram
    plt.subplot2grid((2, 2), (1, 1), colspan=1)
    plt.hist(CV, fill=0,bins=5)
    plt.xlabel('Coefficient of Variation (CV)')
    plt.title('Mean CV: %s' % np.round(np.mean(CV), 3),fontweight='bold')

    plt.tight_layout()
    plt.savefig('../../results/14_poisson_%s.png'%fig_idx,dpi=300)

    ISI_total.append(np.mean(ISI))
    CV_total.append(np.mean(CV))

# plt.show()


plt.subplot(1,2,1)
plt.plot(WEX,ISI_total)
plt.title('ISI vs Synaptic Strength')
plt.xlabel('Synaptic strength')
plt.ylabel('ISI (ms)')

plt.subplot(1,2,2)
plt.plot(WEX,ISI_total)
plt.title('CV vs Synaptic Strength')
plt.xlabel('Synaptic strength')
plt.ylabel('CV')

plt.savefig('../../results/14_poisson_CVISI.png',dpi=300)

# ---------------------------- QUESTIONS ---------------------------- #

# 1 ) Is the spiking output of the neuron irregular with w_e = w_i = 0.5?
# YES it is, since the ISI distribution follows the exponential trend and the CV is found around 1
