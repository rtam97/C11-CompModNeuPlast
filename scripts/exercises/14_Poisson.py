import numpy as np
from matplotlib import pyplot as plt
from scripts.solvers import eulerNeuron as solve
from scripts.neurons import *
from matplotlib import use
from scipy.optimize import curve_fit
# use('Qt5Agg')

from scripts.parameters.param14_poisson import *

from time import time as stopwatch

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



# Euler integration parameters
t_end = 1000        # Simulation time       (ms)
dt = 0.1            # Integration time step (ms)


trials = 10

pot = []
ISI = []
CV = []
SC = []


start = stopwatch()
for trial in range(trials):
    trial_start = stopwatch()
    print('Trial %s / %s'%(trial,trials))
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

# Combine trial voltage and time in a single time series
V_combined = np.concatenate([v for v in pot])
time_combined = np.linspace(0,t_end*trials,len(V_combined))
ISI = np.concatenate(ISI)
CV = np.squeeze(CV)
CV = CV[~np.isnan(CV)]
print('spikes :',np.sum(SC))
print('ISIs   :',len(ISI))
print('CVs   :',len(CV))

# Exponential fit
def exp(x,a,b,c):
    return a*np.exp(-b*x)+c
binz = 20
binval = np.histogram(ISI, bins=binz)[0]
xval = np.linspace(0,np.max(ISI),binz)
popt,pcov = curve_fit(exp,xval,binval)



# Plot combined voltage time series
plt.subplot2grid((2,2),(0,0),colspan=2)
plt.plot(time_combined/1000,V_combined)
plt.xlabel('Time (s)')
plt.ylabel('Memb. potential (mV)')
title = 'Trials %s || Rates (E-I): %s-%s Hz || #Spikes: %s'\
            %(trials,f_ex,f_in,
              np.round(np.sum(SC),3))
plt.title(title)

# Plot ISI histogram
plt.subplot2grid((2, 2), (1, 0), colspan=1)
plt.hist(ISI, fill=0, bins=binz, label='ISI')
plt.plot(xval, exp(xval, *popt), 'r-', label='exp.fit')
plt.legend(loc='upper right')
plt.xlabel('ISI')
plt.ylabel('Frequency')
plt.title('Mean ISI: %s ms '%(np.round(np.mean(ISI),3)))

# Plot CVs histogram
plt.subplot2grid((2, 2), (1, 1), colspan=1)
plt.hist(CV, fill=0)
plt.xlabel('Coefficient of Variation (CV)')
plt.title('Mean CV: %s'%np.round(np.mean(CV),3))

plt.show()

