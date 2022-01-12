# Leaky integrate-and-fire neuron
def LIF(V,E,R,I,tau):
    return (E - V+R*I)/tau
