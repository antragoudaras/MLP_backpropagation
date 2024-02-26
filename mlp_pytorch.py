"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
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
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        super(MLP, self).__init__()
        # Create the network based on the specified hidden sizes

        #template for building the network followed from:
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html
        
        layers = []
        layer_sizes = [n_inputs] + n_hidden
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index-1], layer_sizes[layer_index])]
            if use_batch_norm: # Add batch norm if specified
                layers += [nn.BatchNorm1d(layer_sizes[layer_index])]
            layers += [nn.ELU()]
        layers += [nn.Linear(layer_sizes[-1], n_classes)]
        self.layers = nn.ModuleList(layers) # A module list registers a list of modules as submodules (e.g. for parameters)
        
        #weight initialization following:
        #https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            elif name.startswith("layers.0"):
                param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
            else:
                param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))

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
        out = x.view(x.shape[0], -1) #flatten the image
        for layer in self.layers:
            out = layer(out)
        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
