"""Problem 3 - Training on MNIST"""
import numpy as np


from mytorch.nn.activations import *
from mytorch.nn.batchnorm import BatchNorm1d
from mytorch.nn.linear import Linear
from mytorch.nn.loss import *
from mytorch.nn.sequential import Sequential
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor

# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion

    # TODO: Call training routine (make sure to write it below)
    val_accuracies = None

    model = Sequential(Linear(784, 20), BatchNorm1d(20), ReLU(),
                             Linear(20, 30))
    optimizer = SGD(model.parameters(), lr = 0.1)
    criterion = CrossEntropyLoss()

    val_accuracies = train(model, optimizer, criterion, train_x, train_y, val_x, val_y)

    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    val_accuracies = []
    
    # TODO: Implement me! (Pseudocode on writeup)
    model.train()
    for epoch in range(num_epochs):
        shuffler = np.random.permutation(len(train_y))
        train_x = train_x[shuffler]
        train_y = train_y[shuffler]

        batches = split_data_into_batches(train_x, train_y, 100)
        for i, (batch_data, batch_labels) in enumerate(batches):
            optimizer.zero_grad() # clear any previous gradients
            out = model(Tensor(batch_data))
            loss = criterion(out, Tensor(batch_labels))
            loss.backward()
            optimizer.step() # update weights with new gradients
            if i % 100 == 0:
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                model.train()

    return val_accuracies

def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    #TODO: implement validation based on pseudocode
    model.eval()
    batches = split_data_into_batches(val_x, val_y, 100)
    for (batch_data, batch_labels) in batches:
        out = model(Tensor(batch_data))
        batch_preds = np.argmax(out.data, axis=1)
        num_correct = np.sum(batch_preds == batch_labels)
        accuracy = num_correct / len(val_y)
    return accuracy

def split_data_into_batches(x,y,n):
    batches = []
    for i in range(0, len(x), n):
        # Create an index range for l of n items:
        batches.append((x[i:i+n], y[i:i+n]))
    return batches