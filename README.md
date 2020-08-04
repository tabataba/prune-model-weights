# prune-model-weights
A function that prunes a neural network using pytorch. 

Pruning is to set a subset of weights and biases to zero (and keep them at zero) to produce a sparse neural network with minimal performance loss. 

This is useful for e.g. reducing the size of the neural network, reducing the runtime of the finished neural network.
