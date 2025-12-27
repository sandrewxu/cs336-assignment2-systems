"""
Replication of FlashAttention-2.
"""

from einops import einsum
from jaxtyping import Float
import math
import torch
import triton
import triton.language as tl

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
            Q: Float[torch.Tensor, "... query d_model"] query tensor
            K: Float[torch.Tensor, "... key d_model"]   key tensor
            V: Float[torch.Tensor, "... key d_model"]   value tensor
            is_causal: bool causal mask
        
        Returns:
            O: Float[torch.Tensor, "... query d_model"] output tensor
        """
        # Assert dimensions
        assert K.shape == V.shape, "K and V must have same dimensions"
        assert Q.shape[-1] == V.shape[-1], "Q and V must have the same last dimension"
        batch_size, query, d_model = Q.shape[-3], Q.shape[-2], Q.shape[-1]
        key = K.shape[-2]

        # Tile sizes must be at least 16x16
        ctx.Q_TILE_SIZE = 16    # B_q
        ctx.KV_TILE_SIZE = 16   # B_k

        # Create O and L tensors
        O = torch.empty((batch_size, query, d_model))   # (B_q, d_model)
        L = torch.empty((batch_size, query))    # (B_q,)

        for batch in range(batch_size):
            for i in range(math.ceil(query / ctx.Q_TILE_SIZE)):
                q_start, q_end = i*ctx.Q_TILE_SIZE, (i+1)*ctx.Q_TILE_SIZE
                Q_i = Q[batch, q_start:q_end]  # (B_q, d_model)
                O_i0 = torch.zeros((ctx.Q_TILE_SIZE, d_model))  #(B_q, d_model)
                O_i_prev = O_i0
                l_i0 = torch.zeros((ctx.Q_TILE_SIZE,))   # (B_q,)
                l_i_prev = l_i0
                m_i0 = torch.full((ctx.Q_TILE_SIZE,), -torch.inf)    # (B_q,)
                m_i_prev = m_i0
                for j in range(math.ceil(key / ctx.KV_TILE_SIZE)):
                    # Load K(j), V(j) from global memory
                    kv_start, kv_end = j*ctx.KV_TILE_SIZE, (j+1)*ctx.KV_TILE_SIZE
                    Kj = K[batch, kv_start:kv_end]   # (B_k, d_model)
                    Vj = V[batch, kv_start:kv_end]   # (B_k, d_model)

                    # Compute tile of pre-softmax attention scores
                    S_ij = einsum(Q_i, Kj, "B_q d_model, B_k d_model -> B_q B_k") / math.sqrt(d_model)

                    # Compute the running max tensor
                    row_maxes = torch.max(S_ij, dim=-1).values
                    m_ij = torch.max(torch.stack([row_maxes, m_i_prev], dim=0), dim=0).values   # (B_q, )

                    # Compute softmax estimate
                    P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))   # (B_q, B_k)

                    # Compute the softmax denominator
                    alpha = torch.exp(m_i_prev - m_ij)
                    l_ij = alpha * l_i_prev + torch.sum(P_ij, dim=-1)  # (B_q,)

                    # Compute the output tile
                    O_ij = (alpha.unsqueeze(-1) * O_i_prev) + einsum(P_ij, Vj, "B_q B_k, B_k d_model -> B_q d_model")

                    # Set the m, l, and O values for next iteration
                    m_i_prev = m_ij
                    l_i_prev = l_ij
                    O_i_prev = O_ij

                # Compute O_i and L_i
                O_i = O_i_prev / l_i_prev.unsqueeze(-1) # (B_q, d_model)
                L_i = m_i_prev + torch.log(l_i_prev)    # (B_q,)

                # Write O_i and L_i to global memory
                O[batch, q_start:q_end] = O_i
                L[batch, q_start:q_end] = L_i

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward():
        raise NotImplementedError

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    Fused attention kernel, tile of a single batch.
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, K_TILE_SIZE),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, K_TILE_SIZE),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Loop
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        # Load Q_i from global memory
        Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE, D)
        # Initialize buffers to write to
        O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m = tl.full((Q_TILE_SIZE,), -float("Inf"), dtype=tl.float32)
        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            # Load Kj, Vj from global memory
            Kj = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")   # (K_TILE_SIZE, D)
            Vj = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")   # (K_TILE_SIZE, D)
            # Compute QK^T
            S_ij = tl.dot(Q_i, Kj.T) * scale    # (Q_TILE_SIZE, K_TILE_SIZE)
            # Compute m
            row_maxes = tl.max(S_ij, dim=-1)    # (Q_TILE_SIZE,)
            m_new = tl.max(tl.join(m, row_maxes), dim=-1)   # (Q_TILE_SIZE,)
            # P_ij
            P_ij = tl.exp(S_ij - m_new) # (Q_TILE_SIZE, K_TILE_SIZE)
            # l
            alpha = tl.exp(m - m_new)
            l = alpha * l + tl.sum(P_ij, dim=-1)    # (Q_TILE_SIZE,)
            # O_i
            P_ij.to(Vj.dtype)
            O_i = alpha.unsqueeze(-1) * O_i + tl.dot(P_ij, Vj)
            # Update m, advance pointers
            m = m_new
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # Compute the final values of O_i and l
        O_i = O_i / l.unsqueeze(-1)
        O_i.to(O_block_ptr.type.element_ty)
        l = m + tl.log(l)
        # Write to global memory
        tl.store(O_block_ptr, O_i, boundary_check=(0,))
        tl.store(L_block_ptr, l, boundary_check=(0,))
        # Advance pointers
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))

class FlashAttention2(torch.autograd.Function):
    """
    FlashAttention-2 in Triton.
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
            Q: Float[torch.Tensor, "... query d_model"] query tensor
            K: Float[torch.Tensor, "... key d_model"]   key tensor
            V: Float[torch.Tensor, "... key d_model"]   value tensor
            is_causal: bool causal mask
        
        Returns:
            O: Float[torch.Tensor, "... query d_model"] output tensor
        """
        # Assert dimensions
        assert K.shape == V.shape, "K and V must have same dimensions"
        assert Q.shape[-1] == V.shape[-1], "Q and V must have the same last dimension"
        batch_size, query, d_model = Q.shape[-3], Q.shape[-2], Q.shape[-1]
        key = K.shape[-2]

        # Tile sizes must be at least 16x16
        ctx.Q_TILE_SIZE = 16    # B_q
        ctx.KV_TILE_SIZE = 16   # B_k

        # Create O and L tensors
        O = torch.empty((batch_size, query, d_model))   # (batch, B_q, d_model)
        L = torch.empty((batch_size, query))    # (batch, B_q,)

        # Run kernel with launch grid (T_q, batch_size)
        flash_fwd_kernel[(tl.cdiv(query, ctx.Q_TILE_SIZE, batch_size))](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            query, key,
            1/math.sqrt(d_model),
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward():
        raise NotImplementedError
