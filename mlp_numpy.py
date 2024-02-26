"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        self.layers = []
        first_layer = True # First layer has different weight intialization
        for hidden_layer in n_hidden: 
            # Append a linear layer & activation
            self.layers.append(LinearModule(n_inputs, hidden_layer, first_layer))
            self.layers.append(ELUModule())

            
            n_inputs = hidden_layer # Input of next layer is output of current layer
            first_layer = False # No longer first layer
        if len(n_hidden) == 0: # If no hidden layers, only output layer, multinomial regression
            self.layers.append(LinearModule(n_inputs, n_classes, first_layer))
        else: # Else, last layer is linear layer
            self.layers.append(LinearModule(n_hidden[-1], n_classes, first_layer))
        self.layers.append(SoftMaxModule()) 
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        for layer in self.layers[::-1]: #itereare in reverse orders for backprop
            dout = layer.backward(dout)
    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """

        for layer in self.layers:
            layer.clear_cache()
        