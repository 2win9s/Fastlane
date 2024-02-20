# An arsenal of tricks
In order for this kind of architecture to be at least somewhat useable we need to apply various 'tricks'

## 1. Artificial neural network composed of artificial neural networks
In order to avail of at least some caching and SIMD parallelisation instead of neurons being the building block of the fastlane architecture we can instead opt to use small fixed size neural networks with one input and output. This will allow the same expressivity with less "units" but more overall parameters needed. Though this trade off may be worth it if it can capitalise more on the quirks of modern computing hardware (64 byte cache line, SIMD instructions on small vectors of contingent memory etc.).

Here we will implement the units to build up the fastlane neural networks as simple small neural networks of 1 input 2 hidden layers of 7 units each and 1 output. Thesse numbers seem arbitrary and they are but, they would results in fully connected artificial neural networks containing 111 c++ floats, with float being fp32(on most non edge device implementations) that makes it one float short of perfectly filling 7 cache lines of contingent memory, this is where the exposure to caching efficiency and SIMD parallelisation lies.(The memory usage includes the parameters necessary for implementing Re:Zero, Bachlechner et Al . link:https://arxiv.org/abs/2003.04887)

Then we can treat these artificial neural networks as single neurons for fastlane! In fact if we reframe it, it could be akin to learning a activation function. Of course in general this is redundant, artificial neural networks can approximate virtually any function in a bounded range by the UAT, but doing this allows the fastlane architecture to utilise compute more effectively.

As for the implementation so far the individual component artificial neural networks have 3 activation functions programmed, ReLU, sin and "log_relu" defined as

$f(x) = ln(ReLU(x) + 1)$

Although it introduces significant computational overhead the hope is that the sine and "log_relu" functions can allow for better out of sample performance, capturing periodic properties and interactions beyond approximating at a limited range.

## "Gated State"
We can define a the equation to update hidden states as
note these are not vectors!! This is almost akin to an activation function that takes in the previous hidden state

$h_{t} = h_{t-1} + tanh(a(x_{t}))*\sigma(b(x_{t}))$

This is inspired by LSTM (the gating) and the Selective State Space Models of the MAMBA architecture (only modify state depending on input).

This is necessary as to combat the vanishing gradient problem
To keep this recurrence under control a function such as $a(tanh(x/a)$ can be used to bound it at every timestep.


