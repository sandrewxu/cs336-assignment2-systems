"""
Distributed Data Parallel Training.


"""

import torch
import torch.nn as nn

class DDP(nn.Module):
    """
    Python class to handle distributed data parallel training.
    Takes care of broadcasting weights before training and issuing
    communication calls for gradient averaging.
    """

    def __init__(self, module: nn.Module):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, construct
        a DDP container that will handle gradient synchronization across ranks.
        """

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method with the provided positional
        and keyword arguments.
        """

    def finish_gradient_synchronization(self):
        """
        When called, wait for asynchronous communication calls to be queued on GPU.
        """
