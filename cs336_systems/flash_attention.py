"""
Replication of FlashAttention-2.
"""

from einops import einsum
from jaxtyping import Float
import math
import torch
import triton
import triton.language as tl

@triton.jit
def flashattention_pt_fwd():
    pass

class FlashAttentionPyTorch(torch.autograd.Function):
    """
    FlashAttention-2 in pure PyTorch. This will be slower,
    but implemented for debugging purposes.
    """
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass of a fused attention kernel.
        Returns output O and logsumexp L.
        S = QK^T and P = softmax(S) are not cached.
        L, Q, K, V, O are saved in context

        Args:
            ctx: manager to save dims and L, Q, K, V, O
            Q: Float[torch.Tensor, "query d_model"] query tensor
            K: Float[torch.Tensor, "key d_model"]   key tensor
            V: Float[torch.Tensor, "key d_model"]   value tensor
            is_causal: bool causal mask
        
        Returns:
            O: Float[torch.Tensor, "query d_model"] output tensor
        """
        # Assert dimensions
        assert K.shape == V.shape, "K and V must have same dimensions"
        assert Q.shape[-1] == V.shape[-1], "Q and V must have the same last dimension"
        query, d_model = Q.shape
        key = K.shape[0]

        # Tile sizes must be at least 16x16
        ctx.Q_TILE_SIZE = 16    # B_q
        ctx.KV_TILE_SIZE = 16   # B_k

        # Create O and L tensors
        O = torch.empty((query, d_model))   # (B_q, d_model)
        L = torch.empty((query))    # (B_q,)

        for i in range(1, tl.cdiv(query, ctx.Q_TILE_SIZE) + 1):
            q_start, q_end = i*ctx.Q_TILE_SIZE, (i+1)*ctx.Q_TILE_SIZE
            Q_i = Q[q_start:q_end]  # (B_q, d_model)
            O_i0 = torch.zeros((ctx.Q_TILE_SIZE, d_model))  #(B_q, d_model)
            O_i_prev = O_i0
            l_i0 = torch.zeros((ctx.Q_TILE_SIZE,))   # (B_q,)
            l_i_prev = l_i0
            m_i0 = torch.full((ctx.Q_TILE_SIZE), -torch.inf)    # (B_q,)
            m_i_prev = m_i0
            for j in range(1, tl.cdiv(key, ctx.KV_TILE_SIZE) + 1):
                # Load K(j), V(j) from global memory
                kv_start, kv_end = j*ctx.KV_TILE_SIZE, (j+1)*ctx.KV_TILE_SIZE
                Kj = K[kv_start:kv_end]   # (B_k, d_model)
                Vj = V[kv_start:kv_end]   # (B_k, d_model)

                # Compute tile of pre-softmax attention scores
                S_ij = einsum(Q_i, Kj, "B_q d_model, B_k d_model -> B_q B_k") / math.sqrt(d_model)

                # Compute the running max tensor
                row_maxes = torch.max(S_ij, dim=-1).values
                m_ij = torch.max(torch.stack([row_maxes, m_i_prev], dim=0), dim=0).values   # (B_q, )

                # Compute softmax estimate
                P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))   # (B_q, B_k)

                # Compute the softmax denominator
                l_ij = torch.exp(m_i_prev - m_ij) * l_i_prev + torch.sum(P_ij, dim=-1)  # (B_q,)

                # Compute the output tile
                O_ij = einsum(torch.diag(torch.exp(m_i_prev - m_ij)),  O_i_prev, "B_q B_q, B_q d_model -> B_q d_model") 
                + einsum(P_ij, Vj, "B_q B_k, B_k d_model -> B_q d_model")

                # Set the m, l, and O values for next iteration
                m_i_prev = m_ij
                l_i_prev = l_ij
                O_i_prev = O_ij

            # Compute O_i and L_i
            O_i = einsum(torch.inverse(torch.diag(l_i_prev)), O_i_prev, "B_q B_q, B_q d_model -> B_q d_model")
            L_i = m_i_prev + torch.log(l_i_prev)    # (B_q,)

            # Write O_i and L_i to global memory
            O[q_start:q_end] = O_i
            L[q_start:q_end] = L_i

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward():
        raise NotImplementedError

@triton.jit
def flashattention2_fwd():
    pass

class FlashAttention2(torch.autograd.Function):
    """
    FlashAttention-2 in Triton.
    """
    @staticmethod
    def forward():
        pass

    @staticmethod
    def backward():
        pass
