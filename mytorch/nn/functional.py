import numpy as np

#import mytorch.tensor as tensor
from mytorch import tensor
from mytorch.autograd_engine import Function
import math

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Exp must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        res = np.exp(a.data)
        c = tensor.Tensor(res, requires_grad=requires_grad, is_leaf=not requires_grad) 
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = np.exp(a.data)
        return tensor.Tensor( grad_a * grad_output.data) 


"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        # if a.data.shape != b.data.shape:
        #     raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        """  
        grad_output := dl/dc i.e. gradient of loss w.r.t output (c)
        grad_a := dl/da i.e. gradient of loss w.r.t a
        dl/da = (dl/dc) . (dc/da)
        dc/da = d/da(a + b) ==> 1
        """
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * grad_output.data
        if a.shape != b.shape:
            if a.shape != grad_a.shape:
                grad_a = unbroadcast(grad_a, a.shape)
            if b.shape != grad_b.shape:
                grad_b = unbroadcast(grad_b, b.shape)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'): #or a.data.shape != b.data.shape:
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        #raise Exception("TODO: Implement '-' backward")
        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * grad_output.data * -1
        
        if a.shape != b.shape:
            if a.shape != grad_a.shape:
                grad_a = unbroadcast(grad_a, a.shape)
            if b.shape != grad_b.shape:
                grad_b = unbroadcast(grad_b, b.shape)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

# TODO: Implement more Functions below

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'): #or a.data.shape != b.data.shape:
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data * b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        #raise Exception("TODO: Implement '-' backward")
        # calculate gradient of output w.r.t. each input
        grad_a = b.data * grad_output.data
        grad_b = a.data * grad_output.data

        if a.shape != b.shape:
            if a.shape != grad_a.shape:
                grad_a = unbroadcast(grad_a, a.shape)
            if b.shape != grad_b.shape:
                grad_b = unbroadcast(grad_b, b.shape)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'): #or a.data.shape != b.data.shape:
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data / b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        #raise Exception("TODO: Implement '-' backward")
        # calculate gradient of output w.r.t. each input
        temp_a = (1./b.data)
        grad_a = temp_a * grad_output.data
        temp_b = - (a.data/(b.data**2))
        grad_b = temp_b * grad_output.data

        if a.shape != b.shape:
            if a.shape != grad_a.shape:
                grad_a = unbroadcast(grad_a, a.shape)
            if b.shape != grad_b.shape:
                grad_b = unbroadcast(grad_b, b.shape)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss

    #raise Exception("TODO: Implement XELoss for comp graph")

    p_std = predicted - tensor.Tensor(np.max(predicted.data))
    p_exp = p_std.exp() # predicted.exp()
    p_softmax = p_exp / p_exp.sum(axis=1, keepdims=True)
    p_log_softmax = p_softmax.log()
    targets = to_one_hot(target, num_classes)
    log_lik = targets * p_log_softmax
    log_lik_sum = log_lik.sum(axis=None)
    ce = tensor.Tensor(-1) * log_lik_sum / tensor.Tensor(batch_size)
    return ce


def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]
     
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)

    
def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad



class ReLU(Function):
    @staticmethod
    def forward(ctx, x):

        if not (type(x).__name__ == 'Tensor'): 
            raise Exception("Args must be Tensors: {}".format(type(x).__name__))

        ctx.save_for_backward(x)
        result = np.where(x.data > 0, x.data, np.zeros(x.data.shape))
        requires_grad = x.requires_grad
        z = tensor.Tensor(result, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        grad_x = np.where(x.data >= 0, np.ones(x.data.shape), np.zeros(x.data.shape))
        grad_x = grad_x * grad_output.data
        #grad_x = unbroadcast(grad_x, grad_x.shape[1])
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_x)




class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'): #or a.data.shape != b.data.shape:
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.matmul(a.data, np.transpose(b.data)) , requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        #raise Exception("TODO: Implement '-' backward")
        # calculate gradient of output w.r.t. each input
        grad_a = np.transpose( np.matmul(np.transpose(b.data) , np.transpose(grad_output.data)) )
        grad_b = np.transpose( np.matmul(np.transpose(a.data) , grad_output.data) )
    
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad = np.broadcast_to(grad_output.data, ctx.shape)
        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None



class Power(Function):
    @staticmethod
    def forward(ctx, x, p):

        if not (type(x).__name__ == 'Tensor'): 
            raise Exception("Args must be Tensors: {}".format(type(x).__name__))

        ctx.save_for_backward(x, p)
        result = np.float_power(x.data, p.data)
        requires_grad = x.requires_grad
        z = tensor.Tensor(result, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x,p = ctx.saved_tensors
        grad_x = p.data * np.float_power(x.data, p.data - 1) * grad_output.data
        #grad_p = (p.data - 1) * grad_output.data
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_x), None #tensor.Tensor(grad_p)


class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, is_train=False):
        """Forward pass for dropout layer.

        Args:
            ctx (ContextManager): For saving variables between forward and backward passes.
            x (Tensor): Data tensor to perform dropout on
            p (float): The probability of dropping a neuron output.
                       (i.e. 0.2 -> 20% chance of dropping)
            is_train (bool, optional): If true, then the Dropout module that called this
                                       is in training mode (`<dropout_layer>.is_train == True`).
                                       
                                       Remember that Dropout operates differently during train
                                       and eval mode. During train it drops certain neuron outputs.
                                       During eval, it should NOT drop any outputs and return the input
                                       as is. This will also affect backprop correspondingly.
        """
        if not type(x).__name__ == 'Tensor':
            raise Exception("Only dropout for tensors is supported")
        
        #raise NotImplementedError("TODO: Implement Dropout(Function).forward() for hw1 bonus!")
        
        dropout_val = np.random.binomial(1 ,1-p, size=x.data.shape) * (1.0/(1-p))
        ctx.save_for_backward(x, tensor.Tensor(dropout_val))
        c = x.data
        if is_train:
            c = x.data * dropout_val #* (1.0/(1-p))

        requires_grad = x.requires_grad
        z = tensor.Tensor(c, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return z
        
    @staticmethod
    def backward(ctx, grad_output):
        #raise NotImplementedError("TODO: Implement Dropout(Function).backward() for hw1 bonus!")
        x, dropout_val = ctx.saved_tensors
        grad_a = dropout_val.data * grad_output.data
        return tensor.Tensor(grad_a)

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.
        
        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    #raise NotImplementedError("TODO: Complete functional.get_conv1d_output_size()!")
    return ((input_size - kernel_size)//stride) + 1
    
class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.
        
        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.
        
        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        
        # TODO: Save relevant variables for backward pass
        
        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)

        ctx.save_for_backward(x, weight, tensor.Tensor(output_size), bias, tensor.Tensor(stride))
        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, out_channel, output_size))
        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.

        for i in range(batch_size):
            for j in range(out_channel):
                for k in range(output_size):
                    start = k * stride
                    finish = start + kernel_size
                    out[i,j,k] = np.sum(weight.data[j, :, :] * x.data[i,:,start:finish]) + bias.data[j]
                
        requires_grad = x.requires_grad
        return tensor.Tensor(out, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)

        # TODO: Put output into tensor with correct settings and return 
        # raise NotImplementedError("Implement functional.Conv1d.forward()!")
    
    @staticmethod
    def backward(ctx, grad_output):

        x, weight, output_size, bias, stride = ctx.saved_tensors
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape

        
        dz = grad_output.data #np.asarray([[[1,1],[2,1],[1,2]]]) #np.asarray([[[1,1,1,1],[2,1,2,1],[1,2,1,2]]])  #
        #print('dz:',dz)
        if stride.data > 1:
            dz = upsample(dz, stride.data, input_size, kernel_size)
        dzpad = np.zeros((batch_size, out_channel, (input_size + 2 * (kernel_size - 1) ) ))

        Wflip = weight.data
        for i in range(weight.shape[0]):
            Wflip[i] = np.fliplr(weight.data[i])

        for i in range(batch_size):
            for j in range(out_channel):
                dzpad[i, j, kernel_size-1:input_size ] = dz[i,j,:] #center map

        dy = np.zeros((batch_size, in_channel, input_size))
        for i in range(batch_size):
            for j in range(in_channel):
                for k in range(input_size):
                    segment = dzpad[:,:, k:k+kernel_size]
                    dy[i,j,k] = np.sum(Wflip[:, j, :] * segment[i,:,:]) #np.tensordot(Wflip, segment)

        dw = np.zeros((out_channel, in_channel, kernel_size))
        for i in range(out_channel):
            for j in range(in_channel):
                for k in range(kernel_size):
                    dw[i,j,k] = np.sum(dz[:,i,:] * x.data[ :,j,k:(k+input_size)-(kernel_size-1) ])
        
        grad_data = grad_output.data
        #grad_b = (np.sum(grad_data, axis=2)[0]*grad_data.shape[0])
        grad_b = np.sum(grad_data, axis=(0,2))
        return tensor.Tensor(dy), tensor.Tensor(dw), tensor.Tensor(grad_b), None

def upsample(dz, stride, input_size, kernel_size):
    batch_size, out_channel, out_size = dz.shape
    dzup = np.zeros((batch_size, out_channel, input_size-kernel_size+1 )) 
    for i in range(batch_size):
        for j in range(out_channel):
            for k in range(out_size):
                #dzup[i, j, (k-1) * stride + 1] = dz[i, j, k]
                dzup[i, j, k * stride] = dz[i, j, k]

    return dzup

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensor.Tensor(grad)
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensor.Tensor(grad)


class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        # raise NotImplementedError('Implemented Slice.forward')
        ctx.x = x
        ctx.indices = indices
        requires_grad = x.requires_grad
        sliced = x.data[indices]
        return tensor.Tensor(sliced, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)

    @staticmethod
    def backward(ctx,grad_output):
        #raise NotImplementedError('Implemented Slice.backward')
        res = np.zeros(ctx.x.shape) 
        res[ctx.indices] = grad_output.data
        
        return tensor.Tensor(res)


class Cat(Function):
    @staticmethod
    def forward(ctx,*args):
        '''
        Args:
            args (list): [*seq, dim] 
        
        NOTE: seq (list of tensors) contains the tensors that we wish to concatenate while dim (int) is the dimension along which we want to concatenate 
        '''
        *seq, dim = args
        ctx.seq = seq #[0]
        ctx.dim = dim
        #seq = seq[0]
        concat = seq[0].data
        requires_grad = seq[0].requires_grad
        for i in range(1, len(seq)):
            concat = np.concatenate((concat, seq[i].data), axis = dim)

        return tensor.Tensor(concat, requires_grad=True,
                                           is_leaf=False)
        #raise NotImplementedError('Implement Cat.forward')

    @staticmethod
    def backward(ctx,grad_output):
        #raise NotImplementedError('Implement Cat.backward')

        res = []
        start = 0
        for part in ctx.seq:
            end = start + part.shape[ctx.dim]
            sliced = np.zeros(part.shape)
            
            if ctx.dim == 0:
                sliced_part = grad_output.data[start:end]
            elif ctx.dim == 1:
                sliced_part = grad_output.data[:,start:end]
            elif ctx.dim == 2:
                sliced_part = grad_output.data[:,:,start:end]
            elif ctx.dim == 3:
                sliced_part = grad_output.data[:,:,:,start:end]
            elif ctx.dim == 4:
                sliced_part = grad_output.data[:,:,:,:,start:end]

            sliced += sliced_part

            res.append(tensor.Tensor( sliced ))

            start = end

        return (*res, None)


