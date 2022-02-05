from neurosim.neurosim import Neuron,Stimulus,Simulation
from matplotlib import pyplot as plt
import matplotlib as mlp
# mlp.rcParams['figure.figsize'] = (6,6)
# mlp.use("TkAgg")
dg_sra = 0.06

lif_sra = Neuron(type="lif",
                 Rm = 10,
                 E_k=-70,
                 tau_s=100,
                 sra=dg_sra)

stim = Stimulus(stim_type='constant',
                I_ext=1.45,
                stim_start = 50,
                stim_end = 800,
                t_sim = 1000,
                dt = 0.1,
                neuron=lif_sra)


sim = Simulation(neuron=lif_sra,
                 stimulus=stim)


sim.simulate()





plt.subplot2grid((4,2),(0,0),rowspan=2)
plt.plot(sim.simtime,sim.potential,linewidth=2)
plt.ylabel('Membrane potential (mV)')


plt.subplot2grid((4,2),(2,0),rowspan=1)
plt.plot(sim.simtime,sim.g_sra,linewidth=2,color='tab:purple')
plt.ylabel('g_SRA')

plt.subplot2grid((4,2),(3,0),rowspan=1)
plt.plot(sim.simtime,stim.stim_const,linewidth=2,color='tab:red')
plt.ylabel('Injected current (nA)')
plt.xlabel('time(ms)')
plt.ylim(0,stim.I_ext+1)

plt.subplot2grid((4,2),(0,1),rowspan=4)
plt.plot(sim.outputFreq,linewidth=2,color='tab:orange',marker='D')
plt.xlabel('Spike number')
plt.ylabel('Spike frequency (Hz)')
plt.ylim(20,50)

plt.suptitle('Spike Rate adaptation : dg_SRA = %s '%dg_sra,fontweight='bold')


plt.show()