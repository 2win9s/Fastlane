#an unique architecture for neural networks from scratch
Has similiarities with echo state network or IndRNN, but dumber.

This is a continuation of a futile sci-fest entry that I did with friends during TY (the github repo for the old project has been privated, this is a complete fresh re-write + a few new ideas).

Recurrent neural networks suffer from the vanishing/ exploding gradient problem, that is why LSTM was invented and is used, but why do we need to introduce such a construct? Artificial neural networks are after all universal function approximators can we somehow train a neural network to act as a LSTM unit. Probably needs many more parameters than implementing LSTM and is more computationally wasteful but it is could be an interesting endeavour.

So for the silly idea, "fast lane". 
LSTM retains the information by storing it in the cell state and has the values of the cell state regulated by gates. So what if we all some neurons to be "cell states" i.e. the preactivation z of the the neuron at timestep t is equal to the dot product of weights and state of input neuron (which can be the state at the current timestep or past timestep) plus the previous state,
```math
z_t = f(z_{t-1}) + W \cdot I_{t or t-1}
```
where f is the activation function, W is the slice of the weights matrix corresponding to that neuron and I is the vector of inputs to the neuron. No bias as it is counterproductive, if the state is supposed to remember why try and "erode" it every timestep with a bias?
It probably is possible to have weights and biases that would allow such a neural network to act as an LSTM, but one look at that equation and you know things might just explode as t -> ∞, if we use standard initialisation methods and ReLU or its relatives. 

To add more fuel to the fire, this won't be an ordinary fully connected network. Instead the neurons will be defined by an index sequence in ascending order, and any neuron can take any other neuron as input(the "fastlane" part, every connection can be a skip connection), and the order of activation is sequential. e.g. index 0 has weight connection from index 20 when index 20 is input, index 0 must therefore be taking the state of index 20 at timestep t - 1 as index 20 has a higher index and its state at timestep t hasn't been evaluated yet

the preactivation z of the the neuron at timestep t is equal to the dot product of weights and state input neuron values(which can be the state at the current timestep or past timestep) and the bias,
```math
z_t = b + W \cdot I_{t or t-1}
```
The result is that a neuron will take the value of the input neuron's state at the current time step if it has a higher index than the input, and if it has a lower index it would take the state at the previous time step as input. Also if this network is not fully connected, it infact becomes possible to have some layer arrangement of neurons too, the neurons that don't influence each other's preactivation through a weight can be arranged in a layer (also taking into account the index of the neurons that are inputs). The objective is to allow the neural net to be able to "learn" the number of layers and the number of neurons in each layer through some combination of regularisation of weights and the pruning of weight values close to 0 then rearranging layers according to the remaining weights. This may also allow neural networks to be "patched together", by re indexing the neurons of a neural network and then initialising some new weights.

Since tensorflow, pytorch etc. don't seem to offer the exact functionality I needed and also considering just how slow python is (any performace intestive computation in pure python is a nightmare), this entire project is in C++. Artificial neural networks implemented from scratch (but of course I use std, no boost though). 
Segmentation fault (core dumped)...
