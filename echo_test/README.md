# ECHO (work in progress)

The task of this neural network is to be able to retain the information for a large number of timesteps.
It has 21 neurons in total, the input neurons are the first and second one while the output is the last one.
the "memory" neurons are the 8th 9th and 10th ones. (see the README.md for this repo)

When both of the input neurons are set to 0, the neural net does nothing, if we set the second input neuron to 1 it is now supposed to "record" the value we assign to the first input neuron. Finally when we set the second input neuron to 2 and the first input neuron to 0 the neural net will hopefully "echo" back the remembered value.

So far a few technival issues


note: when compiling the code remember to compile with NN.cpp in from the other folder (and the header file too obviously)
