# Vanishing-intelligence-Exploding-stupidity
a silly idea for an artifical neural network architecture

This is a continuation of a futile sci fest entry that I did with friends during TY (that github repo has been privated, the code is hideous).

Recurrent neural networks suffer from the vanishing/ exploding gradient problem, that is why LSTM was invented and is used, but why do we need to introduce such a construct? Artificial neural networks are after all universal function approximators can we somehow train a neural network to act as a LSTM unit that won't suffer from vanishing/exploding gradients after training it. Who knows it might even have less parameters.

So for the silly idea, "fast lane". LSTM retains the information by storing it in the cell state and has the values of the cell state regulated by gates. So what if we have every neuron be a cell state i.e. the preactivation z of the the neuron at time t is equal to the dor product of weights and inputs neuron values plus the previous state and the bias,
i.e. 
```math
z_t = f(z_{t-1}) + b + W \cdot I
```
where f is the activation function, b is the bias for that neuron, W is the slice of the weights matrix corresponding to that neuron and I is the vector of inputs to the neuron.
It probably is possible to have weights and biases that would allow such a neural network to act as an LSTM, but one look at that equation and you know things might just explode as t->∞, if we use standard initialisation methods and ReLU or its relatives. (perhaps I should experiment with allowing only some neurons to have this property?)

To add more fuel to the fire, this won't be an ordinary fully connected network. Instead the neurons will be defined by an index sequence in ascending order, and any neuron can take any other neuron as input, and the order of activation is sequential, the result is a neuron will take the value of the input neuron's state at the current time step if it has a higher index than the input, and if it has a lower index it would take the state at the previous time step as input. Also if this network is not fully connected, it infact becomes possible to have some layer arrangement of neurons too, the neurons that don't influence each other's preactivation through a weight can be arranged in a layer (also taking into account the index of the neurons that are inputs). The objective is to allow the neural net to be able to "learn" the number of layers and the number of neurons in each layer through some combination of regularisation of weights and the pruning of weight values close to 0 then rearranging layers according to the remaining weights.


