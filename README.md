# Vanishing-intelligence-Exploding-stupidity
a silly idea for an artifical neural network architecture

This is a continuation of a futile sci fest entry that I did with friends during TY (that github repo has been privated, the code is hideous).

Recurrent neural networks suffer from the vanishing/ exploding gradient problem, that is why LSTM was invented and is used, but why do we need to introduce such a construct? Artificial neural networks are after all universal function approximators can we somehow train a neural network to act as a LSTM unit that won't suffer from vanishing/exploding gradients after training it. Who knows it might even have less parameters.

So for the silly idea, "fast lane". LSTM retains the information by storing it in the cell state and has the values of the cell state regulated by gates. So what if we have every neuron be a cell state i.e. the preactivation z of the the neuron at time t is equal to the dor product of weights and inputs neuron values plus the previous state and the bias,
i.e. 
```math
z_t = f(z_t-1) + b + W \cdot I
```
