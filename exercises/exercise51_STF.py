from neurosim.neurosim import Neuron, Stimulus, Simulation
from matplotlib import pyplot as plt
import numpy as np

# -------------------- PARAMETERS -------------------- #
figure = 4  #     2     ,    3     ,     4

if figure == 1:
    stimtype = 'periodic'; U_stf = 0.2; tau_stf = 750; rate = 10; fn = '01'
elif figure == 2:
    stimtype = 'periodic'; U_stf = 0.2; tau_stf = 250; rate = 10; fn = '02'
elif figure == 3:
    stimtype = 'poisson';  U_stf = 0.2; tau_stf = 750; rate = 10; fn = '03'
elif figure == 4:
    stimtype = 'poisson';  U_stf = 0.2; tau_stf = 250; rate = 10; fn = '04'


# -------------------- NEURON -------------------- #
lif = Neuron(type='lif',
             N_exc=10, w_exc=0.0,
             stp='stf', U_stp=U_stf, tau_stf = tau_stf,
             w_fixed=1.0, w_affected=[0,1,2,3,4])

# -------------------- STIMULUS -------------------- #
stim = Stimulus(stim_type=stimtype,rate_exc=10,t_sim=5000,neuron=lif)

# -------------------- SIMULATION -------------------- #
sim = Simulation(neuron=lif,stimulus=stim)
sim.simulate()


# -------------------- PREDICTIONS -------------------- #
# Steady state of U
predicted = U_stf/(1-(1-U_stf)*np.exp(-1/(rate*tau_stf/1000)))
pred_vec = [predicted]*len(sim.stf_u_e[0])

# -------------------- PLOTS -------------------- #
plt.subplot(3,1,1)
plt.plot(sim.simtime/1000,pred_vec,label='predicted U_steady',color='tab:orange')
plt.plot(sim.simtime/1000,sim.stf_u_e[0],color='black',lw=2,label='simulated U')
plt.legend(loc='lower right',fontsize='small')
plt.ylabel('u')
plt.subplot(3,1,2)
plt.plot(sim.simtime/1000,sim.weights_e[0],lw=2)
plt.ylabel('Synaptic Weight')
plt.subplot(3,1,3)
plt.plot(sim.fr_sec,lw=2,color='red')
plt.ylabel('Firing rate (Hz)')
plt.xlabel('Time (s)')
plt.suptitle(f'U_stf : {U_stf} | tau_stf : {tau_stf} ms | stimulus : {stimtype} ({rate} Hz)',fontweight='bold')
# plt.savefig(f'../results/unit5/exercise51_{fn}.png')
plt.show()