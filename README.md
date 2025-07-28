I recently re-implemented Micrograd, Andrej Karpathy's simplified neural network library, from scratch.

Backpropagation, which is done in Micrograd, is at the core of all Deep Learning. It is basically how neural networks "learn".

It's very simple. First the neural network takes your input and does calculations and makes a guess of the output. Then it calculates how wrong it was compared to the correct answer. Finally, it goes back through the neural network and updates the 'weights' based on how much they were responsible for the error. Then it repeats again and again until you're happy with the results.

This project really shed light on the black box of neural networks. I learnt about how gradients flow back in a network and how backpropagation can be applied to any function.

Further to this, I also did backpropagation calculations by hand! I got to feel what it was like being an AI researcher in the 1970s. This really
helped me build intuition for how the gradients flow and how the network updates itself.

Last but not least is the fact that neural networks are actually pretty simple. It's really fascinating to see that with just the chain rule and simple derivatives
we can build something that can learn to approximate any function.

In the Micrograd-Reimplementation.py file I created the Value, Neuron, Layer and MLP class. The Value class is the core class that manages the pointers to the children after every operation 
as well as the backpropagation functions. At the bottom I have done several different examples of backpropagation ranging from a simple function all the way to a tiny dataset using an MLP.
