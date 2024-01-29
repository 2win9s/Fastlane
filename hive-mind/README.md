# Artificial neural networks composed of artificial neural networks (Work in Progress)
In order to avail of at least some caching and SIMD parallelisation instead of neurons being the building block of the fastlane architecture we can instead opt to use small fixed size neural networks with one input and output. This will allow the same expressivity with less "units" but more overall parameters needed. Though this trade off may be worth it if it can capitalise more on the quirks of modern computing hardware (64 byte cache line, SIMD instructions on small vectors of contingent memory etc.).

Here we will implement the units to build up the fastlane neural networks as simple small neural networks of 1 input 2 hidden layers of 7 units each and 1 output. Thesse numbers seem arbitrary and they are but, they would results in fully connected aritficial neural networks containing 111 c++ floats, with float being fp32(on most non edge device implementations) that makes it one float short of perfectly filling 7 cache lines of contingent memory, this is where the exposure to caching efficiency and SIMD parallelisation lies.(The memory usage includes the parameters necessary for implementing Re:Zero, Bachlechner et Al . link:https://arxiv.org/abs/2003.04887)

Then we can treat these artificial neural networks as single neurons for fastlane! In fact if we reframe it, it could be akin to learning a activation function. Of course in general this is redundant, artificial neural networks can approximate virtually any function in a bounded range by the UAT, but doing this allows the fastlane architecture to ustilise compute more effectively.

As for the implementation so far the individual component artificial neural networks have 3 activation functions programmed, ReLU, sin and "log_relu" defined as

$f(x) = ln(ReLU(x) + 1)$

Although it introduces significant computational overhead the hope is that the sine and "log_relu" functions can allow for better out of sample performance, capturing periodic properties and interactions beyond approximating at a limited range.



Of course everything here is untested at the moment, the previous attempts at sentiment analysis were less than great (<50% accuracy at differentiating positive, negative and neutral, see releases). I'm hoping to improve on that here.
