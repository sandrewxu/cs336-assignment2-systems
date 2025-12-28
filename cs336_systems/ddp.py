"""
Distributed Data Parallel Training.


"""

import torch
import torch.distributed as dist
import torch.nn as nn

class DDP_overlap(nn.Module):
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
        super().__init__()
        self.module = module
        self.communication_handles = []
        self.world_size = dist.get_world_size()
        # Broadcast to GPUs; hook parameters; handle tracking
        for p in self.module.parameters():
            with torch.inference_mode():
                dist.broadcast(p, src=0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.hook)

    def hook(self, p: torch.Tensor):
        if p.grad is not None:
            p.grad.div_(self.world_size)
            handle = dist.all_reduce(p.grad, async_op=True)
            self.communication_handles.append(handle)

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method with the provided positional
        and keyword arguments.
        """
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        When called, wait for asynchronous communication calls to be queued on GPU.
        """
        for handle in self.communication_handles:
            handle.wait()
        self.communication_handles = []
