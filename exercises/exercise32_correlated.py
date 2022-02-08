from neurosim.neurosim import Neuron,Stimulus
from neurosim.neurosim import cross_correlogram
from matplotlib import pyplot as plt
from matplotlib import rcParams

directory = '../results/temp/'
title = "Instantaneous Correlation"

# Generate simple LIF neuron with 10 excitatory inputs
lif = Neuron(type='lif',
             N_exc=10)

# ----------------------------------------------------------------- #
# ------------------- INSTANTANEOUS CORRELATION ------------------- #
# ----------------------------------------------------------------- #

# Generate correlated spike trains for Group 1
group1 = Stimulus(stim_type='poisson',
                  rate_exc=10,
                  corr_type='inst',
                  corr=0.3,
                  neuron=lif,
                  t_sim=2000)


# Generate correlated spike trains for Group 2
group2 = Stimulus(stim_type='poisson',
                  rate_exc=10,
                  corr_type='inst',
                  corr=0.3,
                  neuron=lif,
                  t_sim=2000)

# Compute cross-correlogram within Group 1
spike_trains = group1.stim_exc_times
bin_within_group1, height_within_group1 =\
    cross_correlogram(trains=spike_trains, bin_range=[-100,100], dt=5, type='within')

# Compute cross-correlogram within Group 2
spike_trains = group2.stim_exc_times
bin_within_group2, height_within_group2 =\
    cross_correlogram(trains=spike_trains,bin_range=[-100,100],dt=5, type='within')

# Compute cross-correlogram between Group 1 vs Group 2
spike_trains = [group1.stim_exc_times, group2.stim_exc_times]
bin_between, height_between = \
    cross_correlogram(trains=spike_trains, bin_range=[-100,100], dt=5, type='between')


# Megaplot
rcParams['figure.figsize'] = (12,4)

# X-Corr Group 1
plt.subplot(1,4,1)
plt.plot(bin_within_group1,height_within_group1,linewidth=2,label="G1",color='tab:blue')
plt.title('Group 1',fontweight='bold')
plt.xlabel('Spike-lag (ms)')
plt.ylabel('Spike count')

# X-Corr Group 2
plt.subplot(1,4,2)
plt.plot(bin_within_group2,height_within_group2,linewidth=2,label="G2",color='tab:orange')
plt.title('Group 2',fontweight='bold')
plt.xlabel('Spike-lag (ms)')

# X-Corr G1 v G2
plt.subplot(1,4,3)
plt.plot(bin_between,height_between,linewidth=2,label="G1 v G2",color='tab:green')
plt.title('G1 v G2',fontweight='bold')
plt.xlabel('Spike-lag (ms)')

# Toghether
plt.subplot(1,4,4)
plt.plot(bin_within_group1,height_within_group1,linewidth=2,label="G1",color='tab:blue')
plt.plot(bin_within_group2,height_within_group2,linewidth=2,label="G2",color='tab:orange')
plt.plot(bin_between      ,height_between      ,linewidth=2,label="G1 v G2",color='tab:green')
plt.title('cross-correlograms',fontweight='bold')
plt.xlabel('Spike-lag (ms)')
plt.legend(fontsize='small')
plt.suptitle(title,fontweight='bold')

plt.show()

# --------------------------------------------------------------- #
# ------------------- EXPONENTIAL CORRELATION ------------------- #
# --------------------------------------------------------------- #

title = "Exponential Correlation"

# Generate correlated spike trains for Group 1
group1 = Stimulus(stim_type='poisson',
                  rate_exc=10,
                  corr_type='exp',
                  corr=0.3,
                  tau_corr=20,
                  neuron=lif,
                  t_sim=2000)


# Generate correlated spike trains for Group 2
group2 = Stimulus(stim_type='poisson',
                  rate_exc=10,
                  corr_type='exp',
                  corr=0.3,
                  tau_corr=20,
                  neuron=lif,
                  t_sim=2000)


# Compute cross-correlogram within Group 1
spike_trains = group1.stim_exc_times
bin_within_group1, height_within_group1 = \
    cross_correlogram(trains=spike_trains, bin_range=[-100,100], dt=5, type='within')

# Compute cross-correlogram within Group 2
spike_trains = group2.stim_exc_times
bin_within_group2, height_within_group2 = \
    cross_correlogram(trains=spike_trains,bin_range=[-100,100],dt=5, type='within')

# Compute cross-correlogram between Group 1 vs Group 2
spike_trains = [group1.stim_exc_times, group2.stim_exc_times]
bin_between, height_between = \
    cross_correlogram(trains=spike_trains, bin_range=[-100,100], dt=5, type='between')


# Megaplot
rcParams['figure.figsize'] = (12,4)

# X-Corr Group 1
plt.subplot(1,4,1)
plt.plot(bin_within_group1,height_within_group1,linewidth=2,label="G1",color='tab:blue')
plt.title('Group 1',fontweight='bold')
plt.xlabel('Spike-lag (ms)')
plt.ylabel('Spike count')

# X-corr group 2
plt.subplot(1,4,2)
plt.plot(bin_within_group2,height_within_group2,linewidth=2,label="G2",color='tab:orange')
plt.title('Group 2',fontweight='bold')
plt.xlabel('Spike-lag (ms)')

# X-corr G1 v G2
plt.subplot(1,4,3)
plt.plot(bin_between      ,height_between      ,linewidth=2,label="G1 v G2",color='tab:green')
plt.title('G1 v G2',fontweight='bold')
plt.xlabel('Spike-lag (ms)')

# All together now
plt.subplot(1,4,4)
plt.plot(bin_within_group1,height_within_group1,linewidth=2,label="G1",color='tab:blue')
plt.plot(bin_within_group2,height_within_group2,linewidth=2,label="G2",color='tab:orange')
plt.plot(bin_between      ,height_between      ,linewidth=2,label="G1 v G2",color='tab:green')
plt.title('cross-correlograms',fontweight='bold')
plt.xlabel('Spike-lag (ms)')
plt.legend(fontsize='small')
plt.suptitle(title,fontweight='bold')
plt.show()

