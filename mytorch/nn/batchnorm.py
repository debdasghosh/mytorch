from mytorch.tensor import Tensor
import numpy as np
from mytorch.nn.module import Module

class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """
        #raise Exception("TODO!")
        m = x.shape[0]

        # mu_b = (1/m) * np.sum(x.data, axis=0) 
        # x_mu_sq = np.square(x.data - mu_b)
        # var_b = (1/m) * np.sum(x_mu_sq, axis=0) 
        # x_i = (x.data - mu_b) / np.sqrt(var_b + self.eps.data)
        # y_i = self.gamma.data * x_i + self.beta.data

        # sigma_b = (1/(m-1)) * np.sum(x_mu_sq, axis=0) 
        # self.running_mean.data = (1 - self.momentum.data) * self.running_mean.data + self.momentum.data * mu_b
        # self.running_var.data = (1 - self.momentum.data) * self.running_var.data + self.momentum.data * sigma_b
        # return Tensor(y_i)

        if self.is_train:
            one_by_m = Tensor([1/m])
            mu_b = one_by_m * x.sum(axis=0)
            x_mu_sq = (x - mu_b).power(Tensor([2]))
            var_b = one_by_m * x_mu_sq.sum(axis=0) 
            x_i = (x - mu_b) / (var_b + self.eps).power(Tensor([1/2]))
            y_i = self.gamma * x_i + self.beta

            one_by_m_1 = Tensor([1/(m-1)])
            sigma_b = one_by_m_1 * x_mu_sq.sum(axis=0) 
            self.running_mean = (Tensor([1]) - self.momentum) * self.running_mean + self.momentum * mu_b
            self.running_var = (Tensor([1]) - self.momentum) * self.running_var + self.momentum * sigma_b
        else:
            mu_b = self.running_mean
            var_b = self.running_var
            x_i = (x - mu_b) / (var_b + self.eps).power(Tensor([1/2]))
            y_i = self.gamma * x_i + self.beta
        return y_i
        
        
