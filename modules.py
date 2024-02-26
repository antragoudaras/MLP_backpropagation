"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.input_layer = input_layer
        if self.input_layer: #first layer if True
            std = np.sqrt(1/in_features)
        else:
            std = np.sqrt(2/in_features)
            
        self.params['weight'] = np.random.normal(0, std, (in_features, out_features)) # Initialize weights with Kaiming initialization
        self.params['bias'] = np.zeros((out_features, )) # Initialize biases with zeros
        self.grads['weight'] = np.zeros((in_features, out_features)) # Initialize gradients with zeros
        self.grads['bias'] = np.zeros((out_features, )) # Initialize gradients with zeros
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        self.input = x
        #Y = XW + B
        out = np.matmul(x, self.params['weight']) + self.params['bias'] # Forward pass
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        dx = np.matmul(dout, self.params['weight'].T) #dL/dx = dL/dy dy/dh
        self.grads['weight'] = np.matmul(self.input.T, dout) #dL/dw = dL/dy dy/dw
        self.grads['bias'] = np.matmul(np.ones(dout.shape[0]), dout) #dL/db = dL/dy dy/db #summing over first dimension
        return dx
        

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        self.input = None
        self.grads['weight'], self.grads['bias'] = None, None

class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.input = x
        out = np.where(x > 0, x, 1 * (np.exp(x) - 1))
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        dx = np.where(self.input > 0, dout, np.multiply(dout, np.exp(self.input))) #dL/dx = dL/dy dy/dh
        return dx
    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        self.input = None


class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        # Max Trick
        x_max = np.max(x, axis=1)[:, None]
        exp_x = np.exp(x - x_max)
        sum_exp_x = np.sum(exp_x, axis=1)[:, None]

        # Compute softmax output
        out = exp_x / sum_exp_x

        # Save intermediate variables for backward pass
        self.out = out

        return out

    def backward(self, dout):
            """
            Backward pass.
            Args:
                dout: gradients of the previous modul
            Returns:
                dx: gradients with respect to the input of the module

            TODO:
            Implement backward pass of the module.
            """
            
            
            #method 1
            # outer = self.out[:, :, np.newaxis] * self.out[:, np.newaxis, :]
            # outer_einsum = np.einsum('bi,bj->bij', self.out, self.out)
            
    
            # diag = np.zeros((self.out.shape[0], self.out.shape[1], self.out.shape[1]))
            # diag[:, np.arange(self.out.shape[1]), np.arange(self.out.shape[1])] = self.out
          
            # softmax_grad = diag - outer_einsum  #dy/dx

            # # dx_vec = np.matmul(dout[:, np.newaxis, :], softmax_grad).squeeze()
            # dx = np.einsum('bi,bij->bj', dout, softmax_grad)


            #method 2
            # dL/dZ = Y * (dL/dY - (dL/dY * Y)11^T) *-> Hadamard product
            # one_vector = np.ones(dout.shape[1])
            # one_outer = np.outer(one_vector, one_vector)
            one_outer = np.ones((dout.shape[1], dout.shape[1]))
            dx = np.multiply(self.out,\
                             (dout - np.matmul(np.multiply(dout,self.out), one_outer)))
            
            return dx # dL/dx = dL/dy dsoftmax/dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        
        self.out = None

class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """
        
        batch_size = x.shape[0]
        n_classes = x.shape[1]
        y = np.eye(n_classes)[y] 
        loss = np.sum(y * np.log(x), axis=1) #cross entropy loss of all samples 
        out = np.multiply(-(1/batch_size), np.sum(loss)) #total cross entropy loss
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        batch_size = x.shape[0]
        n_classes = x.shape[1]
        y = np.eye(n_classes)[y]  
        dx = np.multiply(-(1/batch_size), np.divide(y, x)) #dL/dx = dL/dy dy/dx 

        return dx