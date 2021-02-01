import sys
import numpy as np

from mytorch.optim.optimizer import Optimizer

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    
    >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    Args:
        params (dict): <some module>.parameters()
        lr (float): learning rate (eta)
        momentum (float): momentum factor (beta)

    Inherits from:
        Optimizer (optim.optimizer.Optimizer)
    """
    def __init__(self, params, lr=0.001, momentum=0.0):
        super().__init__(params) # inits parent with network params
        self.lr = lr
        self.momentum = momentum

        # This tracks the momentum of each weight in each param tensor
        self.momentums = [np.zeros(t.shape) for t in self.params]

    def step(self):
        """Updates params based on gradients stored in param tensors"""
        #raise Exception("TODO: Implement SGD.step()")

        # https://github.com/inessus/CMU_11785_Deep_Learning_Code/blob/master/hw1_autolab/hw1/hw1.py#L143

        if self.momentum == 0.0:
            for k in range(len(self.params)):
                param_k = self.params[k]
                param_k.data = np.asarray(param_k.data) - self.lr * np.asarray(param_k.grad.data)
                self.params[k] = param_k

        else:
            for k in range(len(self.params)):
                self.momentums[k] = - self.lr * self.params[k].grad.data + self.momentums[k] * self.momentum
                self.params[k].data = self.params[k].data + self.momentums[k] 
            


