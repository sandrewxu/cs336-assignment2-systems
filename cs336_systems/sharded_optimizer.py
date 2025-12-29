"""
Optimizer state sharding.
"""

import torch
import torch.distributed as dist
import torch.optim as optim
from typing import Type, Any

class ShardedOptimizer(optim.Optimizer):
    """
    Python class to handle optimizer state sharding.
    Wraps an arbitrary optim.Optimizer and takes care of
    synchronizing updated parameters after each optimizer step.
    """
    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        """
        Initializes the sharded state optimizer. params is a collection
        of parameters to be optimized (or parameter groups, in case the user
        wants to use different hyperparameters, such as learning rates, for 
        different parts of the model); these parameters will be sharded across
        all the ranks. The optimizer_cls parameter specifies the type of optimizer
        to be wrapped (e.g., optim.AdamW). Finally, any remaining keyword arguments 
        are forwarded to the constructor of the optimizer_cls.
        """
        # 1. Get distributed information
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 2. Initialize storage for sharded param groups
        self.rank_to_params = {i: [] for i in range(self.world_size)}
        self.param_to_rank = {}
        defaults = kwargs.copy()
        super().__init__(params, defaults)

        # 3. Construct optimizer
        self.optim = optimizer_cls(self.param_groups, **kwargs)

    def step(self, closure=None, **kwargs):
        """
        Calls the wrapped optimizer's step() method with the
        provided closure and keyword arguments. After updating
        the parameters, synchronize with the other ranks.
        """
        self.optim.step(closure=closure, **kwargs)

        # Broadcast each param from rank to all other ranks
        for rank in range(self.world_size):
            params_to_broadcast = self.rank_to_params.get(rank, [])
            for param in params_to_broadcast:
                dist.broadcast(param.data, src=rank, async_op=False)

    def add_param_group(self, param_group: dict[str, Any]):
        """
        Add a parameter group to the sharded optimizer.
        This is called during the construction of the optimizer
        by the super-class constructor and may also be called
        during training (e.g., for gradually unfreezing layers
        in a model). This method handles assigning the model's
        parameters among the ranks.
        """
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = list(params)

        # Deduplicate parameters (handle tied weights)
        seen = set()
        unique_params = []
        for param in params:
            param_id = id(param)
            if param_id not in seen:
                seen.add(param_id)
                unique_params.append(param)

        my_params = []
        for _, param in enumerate(params):
            param_id = id(param)

            # If this parameter hasn't been assigned yet, assign it
            if param_id not in self.param_to_rank:
                owner_rank = len(self.param_to_rank) % self.world_size
                self.param_to_rank[param_id] = owner_rank
                self.rank_to_params[owner_rank].append(param)
                if owner_rank == self.rank:
                    my_params.append(param)
            else:
                # Parameter already assigned, check if this rank owns it
                owner_rank = self.param_to_rank[param_id]
                if owner_rank == self.rank:
                    my_params.append(param)

        if my_params:
            new_param_group = {k: v for k, v, in param_group.items() if k != "params"}
            new_param_group["params"] = my_params
            super().add_param_group(new_param_group)
