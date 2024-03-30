# An unique artificial neural nework architecture from scratch
NN file and Implementation is previous attempt, new implementation in folder hivemind under the headerfile versionX.hpp, e.g. X = 0.


This is a continuation of a futile sci-fest entry that I did with friends during transition year in secondary school (the github repo for the old project has been privated, this is a complete fresh re-write + a few new ideas).

So instead of an ordinary "layer defined" network, the neurons will be defined by an index sequence in ascending order, and any neuron can take any other neuron as input(the "fastlane" part, every connection can be a skip connection), and the order of activation is sequential. 

the preactivation z of the the neuron at timestep t is equal to the dot product of weights and state input neuron values(which can be the state at the current timestep) and the bias,
```math
z_t = b + W \cdot I_{t}
```

The result is that a neuron will take the value of the input neuron's state at the current time step if it has a higher index than the input, and if it has a lower index it would take the state at the previous time step as input. Also if this network is not fully connected, it in fact becomes possible to have some layer arrangement of neurons, the neurons that don't influence each other's preactivation through a weight in the present timestep can be arranged in a layer (also taking into account the index of the neurons that are inputs). 

The objective is to allow the neural net to be able to "learn" the number of layers and the number of neurons in each layer through some combination of regularisation of weights and the pruning of weight values close to 0 then rearranging layers according to the remaining weights. This also allows multiple neural networks to be "patched together", by re-indexing the neurons of a neural network and then initialising some new weights.

An obvious issue is neurons are not fixed to a certain layer, can end up in different ones at different points during training, or even have new neurons introduced. As a consequence popular normalisation methods that enable the training of very deep neural networks such as batchnorm and layernorm cannot be applied. Thus ReZero (Bachlechner et al 2020) is used, as it is a mostly architecture agnostic method that seems to work.

The artificial neural networks are implemented from scratch i.e. no use of ml libraries.
