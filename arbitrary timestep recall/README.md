# ECHO 

A 300+ parameter model that has been trained to be able to record recall and forget a floating point integer with decent accuracy over arbitrary? timesteps(only tested across 10000).
No gates or LSTM needed, just many more parameters, and countless hours of lost sanity trying to train it. Probably performs much worse too

Input neurons are at index 0,1 and 2 the output neuron is indexed at 21, and the special "memory" neuron that retains
its previous state is indexed at 11. When the input is {x,1,0} the network will "remember" x, when the input is {0,2,0} the output of the neural network will be x. To forget x the input {0,0,6} is fed to the network. In all other situations 
{0,0,0} is given as input and the neural net does nothing.

The neural net is serialised as the .txt file and must be present for anything to work. Don't forget to compile (if you want) with the headerfile and implentation, NN.cpp and NN.hpp

This demonstrates that this kind of neural network is indeed trainable (but very finicky), and can also recall information for seemingly arbitrary timesteps (not mathematically proven).


None of the code is parallelised because it is pointless for such a small model and the overhead isn't worth it
