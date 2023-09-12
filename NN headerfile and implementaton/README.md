# FASTLANE

This my C++ implementation of "fastlane" (a trainable echo state network sort of) and the associated functions encapsulated in a struct, it is quite bare bones and the serialization/ deserialization to text is in a terrible format. If for whatever reason you stumble accross it and want to play with it, I suggest reading the implementation and adding what you want. It is quite short and has been constantly changing so I'm not writing documentation just yet, hopefully the comments are suffcient.

note, some of the "#pragma omp simd" are ignored by the compiler, I litter them everywhere with no regard whether it is actually possible or not
