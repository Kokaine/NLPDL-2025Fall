import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, List, Tuple
from einops import einsum, rearrange

class Linear(nn.Module):
    """Applies a linear transformation to the input: y = xA^T + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the linear module.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If True, includes a bias term. Defaults to False.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        std_n = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std_n, a=-3*std_n, b=3*std_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """

        output = x @ self.weight.T

        if self.bias is not None:
            output = output + self.bias
        
        return output


class Embedding(nn.Module):
    """A lookup table that maps indices to embedding vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        # Initialize parameters
        std_n = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std_n, a=-3*std_n, b=3*std_n)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Looks up embedding vectors for token IDs.

        Args:
            token_ids (torch.Tensor): Input tensor of shape (...).

        Returns:
            torch.Tensor: Output tensor of shape (..., embedding_dim).
        """

        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """Applies Root Mean Square Layer Normalization (RMSNorm)."""  

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """

        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMSNorm to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight

        return result.to(in_dtype)

class SwiGLU(nn.Module):
    """Applies the SwiGLU feedforward transformation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the SwiGLU module.

        Args:
            d_model (int): Hidden dimension of the model.
            d_ff (int): Inner dimension of the feedforward layer.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()

        self.w1 = Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SwiGLU transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """

        gate = self.w1(x) 
        data = self.w3(x)

        activated_gate = gate * torch.sigmoid(gate)
        gated_data = activated_gate * data
        output = self.w2(gated_data)

        return output

class RoPE(nn.Module):
    """Applies Rotary Position Embeddings (RoPE)."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initializes the RoPE module.

        Args:
            theta (float): Θ value for the rotary embedding.
            d_k (int): Dimension of query and key vectors.
            max_seq_len (int): Maximum sequence length supported.
            device (torch.device, optional): Device to store buffers. Defaults to None.
        """
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even for RoPE."

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device

        theta_power = torch.arange(0, self.d_k, 2, device=self.device) / self.d_k
        theta_k_freqs = 1.0 / self.theta ** theta_power
        position_ids = torch.arange(self.max_seq_len, device=self.device).unsqueeze(1)
        idx_theta = torch.einsum('i,j->ij', torch.arange(self.max_seq_len, device=self.device), theta_k_freqs)

        cos_cache = torch.cos(idx_theta)
        sin_cache = torch.sin(idx_theta)

        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Applies rotary position embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of shape (..., seq_len)
                specifying token positions.

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_k).
        """
        cos_pos = self.cos_cache[token_positions]
        sin_pos = self.sin_cache[token_positions]

        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([x1 * cos_pos - x2 * sin_pos,
                                 x1 * sin_pos + x2 * cos_pos], dim=-1)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax activation function.

    Applies the softmax function to the input tensor along the specified dimension.

    Args:
    x: Input tensor.
    dim: Dimension along which softmax will be computed. Defaults to -1.

    Returns:
    Tensor with softmax applied along the specified dimension.
    """
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Scaled dot-product attention function.

    Args:
        query: Tensor of shape (batch_size, ..., seq_len_q, d_k)
        key: Tensor of shape (batch_size, ..., seq_len_k, d_k)  
        value: Tensor of shape (batch_size, ..., seq_len_v, d_v)
        mask: Boolean tensor of shape (seq_len_q, seq_len_k) or broadcastable shape

    Returns:
        Tensor of shape (batch_size, ..., seq_len_q, d_v)
    """

    d_k = query.size(-1)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

    attn_weight = softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weight, value)

    return output

class CasualMultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rope: bool = False,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        """Initializes the attention module.

        Args:
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            device (torch.device, optional): Device to store parameters. Defaults to None.
        dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE. Defaults to False.
            theta (float, optional): Θ parameter for RoPE when enabled. Defaults to None.
            max_seq_len (int, optional): Maximum sequence length for RoPE buffers.
                Defaults to None.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.q_linear = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_linear = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_linear = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.out_linear = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

        if self.use_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Applies causal multi-head self-attention.

        Args:
        x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).
            token_positions (torch.Tensor, optional): Tensor of shape (..., seq_len)
                with token positions; required if `use_rope` is True. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        batch_shape = x.shape[:-2]
        seq_len = x.size(-2)

        Q = self.q_linear(x).view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = self.k_linear(x).view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = self.v_linear(x).view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

        if self.use_rope:
            token_pos = token_positions.unsqueeze(-2)
            Q = self.rope(Q, token_positions=token_pos)
            K = self.rope(K, token_positions=token_pos)

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, seq_len, self.d_model)
        output = self.out_linear(attn_output)

        return output

class TransformerBlock(nn.Module):
    """A single Transformer block with self-attention and feedforward network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rope: bool = False,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        """Initializes the Transformer block.

        Args:
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden dimension of the feedforward layer.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE in self-attention. Defaults to False.
            theta (float, optional): Θ parameter for RoPE. Defaults to None.
            max_seq_len (int, optional): Maximum sequence length for RoPE buffers. Defaults to None.
        """
        super().__init__()

        self.pre_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CasualMultiheadSelfAttention(
            d_model,
            num_heads,
            device=device,
            dtype=dtype,
            use_rope=use_rope,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        self.post_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        bs, seq_len, _ = x.shape
        token_pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bs, -1)
        x = x + self.attn(self.pre_norm(x), token_positions=token_pos)
        x = x + self.ffn(self.post_norm(x))
        return x

class TransformerLM(nn.Module):
    """A Transformer-based language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rope: bool = False,
        theta: Optional[float] = None,
    ) -> None:
        """Initializes the Transformer language model.

        Args:
            vocab_size (int): Vocabulary size for token embeddings.
            context_length (int): Maximum sequence length for positional encodings.
            num_layers (int): Number of Transformer blocks.
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden dimension of the feedforward layer.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE. Defaults to False.
            theta (float, optional): Θ parameter for RoPE. Defaults to None.
        """
        super().__init__()

        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                device=device,
                dtype=dtype,
                use_rope=use_rope,
                theta=theta,
                max_seq_len=context_length,
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Applies the Transformer language model.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (..., seq_len).

        Returns:
            torch.Tensor: Logits of shape (..., seq_len, vocab_size).
        """
        x = self.token_embedding(input_ids)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.ln_final(x)
        logits = self.output_linear(x) 

        return logits

class LSTMCell(nn.Module):
    """A single Long Short-Term Memory (LSTM) cell."""

    def __init__(
        self,
        d_model: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the LSTM cell.

        Args:
            d_model (int): Hidden dimension of the LSTM.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LSTM cell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_model).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_state, cell_state), each of shape (batch_size, d_model).
                If None, both are initialized to zeros. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The next (hidden_state, cell_state),
            each of shape (batch_size, d_model).
        """
        ...

class LSTM(nn.Module):
    """Multi-layer LSTM network with batch-first input."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the multi-layer LSTM.

        Args:
            d_model (int): Hidden dimension of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Applies the multi-layer LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_states, cell_states), each of shape
                (num_layers, batch_size, d_model). Defaults to None.

        Returns:
            tuple:
                - torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
                - tuple[torch.Tensor, torch.Tensor]: Next (hidden_states, cell_states),
                    each of shape (num_layers, batch_size, d_model).
        """
        ...

class LSTMLM(nn.Module):
    """LSTM-based language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        """Initializes the LSTM language model.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Hidden dimension of the LSTM.
            num_layers (int): Number of LSTM layers.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Applies the LSTM language model.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_states, cell_states), each of shape
                (num_layers, batch_size, d_model). Defaults to None.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        ...