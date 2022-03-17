#%%
from neurosim.neurosim import Neuron, Stimulus, Simulation
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import use
import numpy as np
# use('tkagg')
prop = 1.2
rcParams['figure.figsize'] = (prop*6.4,prop*4.8)

#%%
# -------------------- PARAMETERS -------------------- #

fignum = 4  # 1 2 3 4

for fignum in [1, 2, 3, 4]:
    rates = [1, 20]

    if fignum == 1:
        fn = '01'; stimtype = 'periodic' ; tau_stf  = 750; tau_std  = 50
    elif fignum == 2:
        fn = '02'; stimtype = 'periodic' ; tau_stf  = 50 ; tau_std  = 750
    elif fignum == 3:
        fn = '03'; stimtype = 'poisson' ; tau_stf  = 750 ; tau_std  = 50
    elif fignum == 4:
        fn = '04'; stimtype = 'poisson' ; tau_stf  = 50 ; tau_std  = 750


    transmission_ratio = [0]*len(rates)
    U_stp = 0.45
    w_fixed = 5


    for i,rate in enumerate(rates):

        if tau_stf < tau_std:
            titl = f'STD Dominated | tau_stf:tau_std {tau_stf}:{tau_std} ms | {stimtype} ({rate} Hz)'
        else:
            titl = f'STF Dominated | tau_stf:tau_std {tau_stf}:{tau_std} ms | {stimtype} ({rate} Hz)'


        # -------------------- NEURON -------------------- #
        lif = Neuron(type='lif',
                     N_exc=5, w_exc=2.0,
                     stp='both',
                     U_stp=U_stp, tau_stf = tau_stf, tau_std=tau_std,
                     w_fixed=w_fixed)

        # -------------------- STIMULUS -------------------- #
        stim = Stimulus(stim_type=stimtype,rate_exc=rate,t_sim=5000,neuron=lif)

        # -------------------- SIMULATION -------------------- #
        sim = Simulation(neuron=lif,stimulus=stim)
        sim.simulate()


        # -------------------- SPIKE TRANSMISSION -------------------- #
        transmission_ratio[i] = np.round(sim.spikeCount/sim.pre_spike_count_e[0],5)[0]
        print('\n',transmission_ratio[i])

        # -------------------- PLOTS -------------------- #
        plt.subplot(3,1,1)
        plt.plot(sim.simtime/1000,sim.stf_u_e[0],color='k',lw=2,label='simulated U')
        plt.xticks([])
        plt.title('STF Dynamics',fontweight='bold')
        plt.ylabel('u(t)',fontweight='bold')
        plt.subplot(3,1,2)
        plt.plot(sim.simtime/1000,sim.std_x_e[0],color='tab:cyan',lw=2,label='simulated U')
        plt.ylabel('x(t)',fontweight='bold')
        plt.xticks([])
        plt.title('STD Dynamics',fontweight='bold')
        plt.subplot(3,1,3)
        plt.plot(sim.simtime/1000,sim.weights_e[0],lw=2)
        plt.title(f'Transmission ratio : {transmission_ratio[i]}',fontweight='bold')
        plt.ylabel('Synaptic Weight',fontweight='bold')
        plt.xlabel('Time (s)',fontweight='bold')
        plt.suptitle(titl,fontweight='bold')
        plt.savefig(f'../results/unit5/exercise52_{fn}_{rate}hz_b.png')
        plt.show()
