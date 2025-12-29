"""
Distributed Data Parallel Training.


"""

import copy
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

class DDP_bucketed(nn.Module):
    """
    Python class to handle distributed data parallel training,
    using gradient bucketing to improve communication efficiency.
    Takes care of broadcasting weights before training and issuing
    bucketed communication calls for gradient averaging.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, construct
        a DDP container that will handle gradient synchronization across ranks.
        """
        super().__init__()
        self.module = module
        self.communication_handles = []
        self.world_size = dist.get_world_size()

        # Create parameter buckets
        buckets = [[]]
        running_size = 0
        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue
            buckets[-1].append(p)
            running_size += (p.numel() * p.element_size()) / (1024 * 1024)
            if running_size > bucket_size_mb:
                buckets.append([])
                running_size = 0

        if not buckets[-1]:
            buckets.pop()

        # Iterate through buckets
        self.param_to_bucket_id = {}
        self.param_to_grad_view = {}
        self.bucket_buffers = []
        self.bucket_grad_buffers = []
        self.bucket_total_counts = []
        self.bucket_remaining_counts = []
        for (i, bucket_params) in enumerate(buckets):
            flat_bucket = torch._utils._flatten_dense_tensors(bucket_params)
            flat_grad_bucket = torch.zeros_like(flat_bucket)
            views = torch._utils._unflatten_dense_tensors(flat_bucket, bucket_params)
            grad_views = torch._utils._unflatten_dense_tensors(flat_grad_bucket, bucket_params)

            for p, v, g_v in zip(bucket_params, views, grad_views):
                self.param_to_bucket_id[p] = i
                self.param_to_grad_view[p] = g_v
                p.data = v  # link param data to bucket
                p.grad = g_v
                p.register_post_accumulate_grad_hook(self.hook)

            self.bucket_buffers.append(flat_bucket)
            self.bucket_grad_buffers.append(flat_grad_bucket)
            self.bucket_total_counts.append(len(bucket_params))
            self.bucket_remaining_counts.append(len(bucket_params))

        for p in self.module.parameters():
            with torch.inference_mode():
                dist.broadcast(p, src=0)

    def hook(self, p: torch.Tensor):
        bucket_id = self.param_to_bucket_id[p]
        grad_view = self.param_to_grad_view[p]

        if p.grad is not None:
            grad_view.copy_(p.grad)

        self.bucket_remaining_counts[bucket_id] -= 1

        if self.bucket_remaining_counts[bucket_id] == 0:
            grad_buffer = self.bucket_grad_buffers[bucket_id]
            grad_buffer.div_(self.world_size)
            handle = dist.all_reduce(grad_buffer, async_op=True)
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
        for p, grad_view in self.param_to_grad_view.items():
            if p.requires_grad:
                p.grad = grad_view
        self.bucket_remaining_counts = copy.deepcopy(self.bucket_total_counts)
