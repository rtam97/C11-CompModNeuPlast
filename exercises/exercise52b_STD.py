#%%
from neurosim.neurosim import Neuron, Stimulus, Simulation
from matplotlib import pyplot as plt

from matplotlib import use

import numpy as np

use('tkagg')

#%%
# -------------------- PARAMETERS -------------------- #

fignum = 6  # 5 6
U_stp = 0.45
w_fixed = 5.0

if fignum == 5:
    fn = '05'; stimtype = 'periodic'
    STF  = [750, 50]
    STD  = [50, 750]
    trials = 1


elif fignum == 6:
    fn = '06'; stimtype = 'poisson'
    STF  = [750, 50]
    STD  = [50, 750]
    trials = 25

rates = [1,5,10,15,20]
stp_ratios = [0, 1]
transmission_ratio = np.zeros((trials,len(rates)))

for r in stp_ratios:
    print('ratio',r)
    for t in range(trials):
        print(f'trial {t+1} / {trials}')
        for i,rate in enumerate(rates):
            print(i+1,len(rates),rate)
            if fignum >= 5 :
                tau_std = STD[r]
                tau_stf = STF[r]


            # -------------------- NEURON -------------------- #
            lif = Neuron(type='lif',
                         N_exc=5, w_exc=2.5,
                         stp='both',
                         U_stp=U_stp, tau_stf = tau_stf, tau_std=tau_std,
                         w_fixed=w_fixed)

            # -------------------- STIMULUS -------------------- #
            stim = Stimulus(stim_type=stimtype,rate_exc=rate,t_sim=10000,neuron=lif)

            # -------------------- SIMULATION -------------------- #
            sim = Simulation(neuron=lif,stimulus=stim)
            sim.simulate(verbose=False)


            # -------------------- SPIKE TRANSMISSION -------------------- #
            transmission_ratio[t][i] = np.round(sim.spikeCount/sim.pre_spike_count_e[0],5)[0]
            # print('\n',transmission_ratio[t][i])

    mu = np.mean(transmission_ratio,axis=0)
    std = np.std(transmission_ratio,axis=0)

    if tau_stf > tau_std:
        lab = f'STF Dominated'
    else:
        lab = f'STD Dominated'

    plt.errorbar(rates,mu/max(mu),lw=2,label=lab,marker='o',yerr=std/max(std))
    plt.xlabel('Stimulus Frequency (Hz)')
    plt.ylabel('Spike transmission ratio')
    plt.grid(visible=1)
    plt.title(f'{stimtype}',fontweight='bold')

plt.legend()
# plt.savefig(f'../results/unit5/exercise52_{fn}_.png')

plt.show()