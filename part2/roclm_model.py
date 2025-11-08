"""
RocLM Implementation - Educational Version

This implementation provides a transformer-based language model
with advanced attention mechanisms and efficient architecture.

Key Components for Students to Implement:
1. RocLMRMSNorm: Root Mean Square Normalization (simpler than LayerNorm)
2. RocLMMLP: Multi-Layer Perceptron with SwiGLU activation
3. RocLMAttention: Multi-head attention with Grouped Query Attention (GQA)

Architecture Overview:
- Transformer Decoder: Autoregressive language model
- RoPE: Rotary Position Embedding for positional encoding
- GQA: Grouped Query Attention for efficiency
- SwiGLU: Gated activation function in MLP
- RMSNorm: Efficient normalization technique

Learning Objectives:
- Understand transformer architecture components
- Learn about modern attention mechanisms (GQA)
- Implement efficient normalization techniques
- Work with gated activation functions
- Understand positional encoding methods
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RocLMConfig:
    """RocLM configuration for the language model"""

    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768

    # RoPE settings
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None

    # Normalization
    rms_norm_eps: float = 1e-6

    # Activation
    hidden_act: str = "silu"

    # Attention settings
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Model settings
    pad_token_id: Optional[int] = None
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    tie_word_embeddings: bool = True

    # Layer types (simplified - all full attention for now)
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.layer_types is None:
            # All layers use full attention for simplicity
            self.layer_types = ["full_attention"] * self.num_hidden_layers


class RocLMRMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Normalization) implementation for RocLM
    
    RMSNorm is a normalization technique that normalizes inputs by their root mean square.
    It's simpler and more efficient than LayerNorm while providing similar benefits.
    
    Key differences from LayerNorm:
    - LayerNorm: normalizes by mean and variance
    - RMSNorm: normalizes only by root mean square (no mean centering)
    
    Mathematical formula:
    RMSNorm(x) = weight * x / sqrt(mean(x^2) + epsilon)
    
    Why use RMSNorm?
    - More efficient (no mean computation)
    - Often performs as well as LayerNorm
    - Reduces computational overhead in large models
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Initialize the learnable weight parameter
        # Hint: This should be a parameter that scales the normalized values
        # The weight should have the same size as the hidden dimension
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Learnable scaling parameter
        self.variance_epsilon = eps  # Small constant to prevent division by zero

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm
        
        Steps:
        1. Store original dtype for later conversion back
        2. Convert to float32 for numerical stability
        3. Calculate the root mean square of the input
        4. Normalize by dividing by RMS
        5. Scale by learnable weight
        6. Convert back to original dtype
        
        Args:
            hidden_states: Input tensor of shape (..., hidden_size)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Store original dtype to convert back later
        input_dtype = hidden_states.dtype
        
        # Convert to float32 for numerical stability during normalization
        hidden_states = hidden_states.to(torch.float32)
        
        # TODO: Calculate the variance (mean of squared values)
        # Hint: Use .pow(2) to square, then .mean(-1, keepdim=True) to average over last dimension
        # The last dimension is the feature dimension (hidden_size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        
        # TODO: Normalize the hidden states
        # Hint: Divide by sqrt(variance + epsilon) using torch.rsqrt for efficiency
        # rsqrt(x) = 1/sqrt(x), which is more efficient than 1/sqrt(x)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # TODO: Apply learnable scaling
        # Hint: Multiply by the weight parameter and convert back to original dtype
        return self.weight * hidden_states.to(input_dtype)


class RocLMMLP(nn.Module):
    """
    MLP (Multi-Layer Perceptron) implementation for RocLM using SwiGLU activation
    
    This MLP uses a "gated" architecture with two parallel linear transformations:
    - gate_proj: Projects input to intermediate size (acts as a "gate")
    - up_proj: Projects input to intermediate size (acts as the main transformation)
    
    The SwiGLU activation combines these as: SwiGLU(x) = Swish(gate(x)) * up(x)
    where Swish(x) = x * sigmoid(x) (also called SiLU)
    
    Architecture:
    Input (hidden_size) -> [gate_proj, up_proj] -> Intermediate (intermediate_size)
    -> SwiGLU activation -> down_proj -> Output (hidden_size)
    
    Why SwiGLU?
    - More expressive than standard ReLU/GELU
    - Gating mechanism allows selective information flow
    - Often performs better in transformer models
    - intermediate_size is typically 4x larger than hidden_size
    """

    def __init__(self, config: RocLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Initialize the three linear projections
        # Hint: gate_proj and up_proj both map from hidden_size to intermediate_size
        # down_proj maps from intermediate_size back to hidden_size
        # All projections typically use bias=False for efficiency
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # Gate projection
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)    # Up projection  
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # Down projection

        # TODO: Choose activation function based on config
        # Hint: Check config.hidden_act and set self.act_fn accordingly
        # Common options: "silu" (SwiGLU), "gelu", "relu"
        if config.hidden_act == "silu":
            self.act_fn = F.silu  # SiLU/Swish: x * sigmoid(x)
        else:
            self.act_fn = F.gelu  # GELU: x * Φ(x) where Φ is standard normal CDF

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU MLP
        
        Mathematical formula:
        SwiGLU(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))
        
        Steps:
        1. Apply gate projection and activation
        2. Apply up projection  
        3. Element-wise multiply (gating)
        4. Apply down projection
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # TODO: Implement the SwiGLU forward pass
        # Hint: 
        # 1. Apply gate_proj to x, then apply activation function
        # 2. Apply up_proj to x
        # 3. Element-wise multiply the results (this is the "gating")
        # 4. Apply down_proj to get final output
        # 
        # The gating mechanism: activated_gate * up_projection
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input for RoPE.
    
    This is a key component of Rotary Position Embedding (RoPE).
    It splits the input tensor in half and rotates the second half by 180 degrees.
    
    Mathematical operation:
    - Split x into two halves: x1 = x[..., :d//2], x2 = x[..., d//2:]
    - Return: [-x2, x1] (concatenate -x2 and x1)
    
    This rotation helps encode positional information in the attention mechanism.
    
    Args:
        x: Input tensor of shape (..., d) where d is even
        
    Returns:
        Rotated tensor of same shape as input
    """
    x1 = x[..., : x.shape[-1] // 2]  # First half
    x2 = x[..., x.shape[-1] // 2 :]  # Second half
    return torch.cat((-x2, x1), dim=-1)  # Rotate: [-x2, x1]


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) to query and key tensors.
    
    RoPE is a positional encoding method that rotates query and key vectors
    based on their positions. This allows the model to understand relative
    positions between tokens.
    
    Mathematical formula:
    RoPE(x, m) = x * cos(mθ) + rotate_half(x) * sin(mθ)
    
    Where:
    - x is the input vector (query or key)
    - m is the position
    - θ is a frequency parameter
    
    This is more efficient than absolute positional embeddings and
    naturally handles variable sequence lengths.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine values for positions
        sin: Sine values for positions
        
    Returns:
        Tuple of (rotated_query, rotated_key) with same shapes as inputs
    """
    cos = cos.unsqueeze(1)  # Add head dimension for broadcasting
    sin = sin.unsqueeze(1)  # Add head dimension for broadcasting
    
    # Apply RoPE: x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats key-value states for Grouped Query Attention (GQA).
    
    In GQA, we have fewer key-value heads than query heads. This function
    repeats each key-value head to match the number of query heads.
    
    For example, if we have 8 query heads and 2 key-value heads:
    - Each key-value head is repeated 4 times (8/2 = 4)
    - This allows each group of 4 query heads to share the same key-value pair
    
    Args:
        hidden_states: Key or value tensor of shape (batch, num_key_value_heads, seq_len, head_dim)
        n_rep: Number of repetitions (num_attention_heads / num_key_value_heads)
        
    Returns:
        Repeated tensor of shape (batch, num_attention_heads, seq_len, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    # If no repetition needed, return as is
    if n_rep == 1:
        return hidden_states
    
    # Repeat each key-value head n_rep times
    # Shape: (batch, num_key_value_heads, 1, seq_len, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    
    # Reshape to combine repeated heads
    # Shape: (batch, num_key_value_heads * n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RocLMRotaryEmbedding(nn.Module):
    """Rotary Position Embedding implementation for RocLM"""

    def __init__(self, config: RocLMConfig, device=None):
        super().__init__()
        self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # Create inverse frequencies
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.int64).float()
                / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RocLMAttention(nn.Module):
    """
    Multi-headed attention for RocLM with Grouped Query Attention (GQA)
    
    This attention mechanism uses Grouped Query Attention (GQA) which is more efficient
    than standard multi-head attention while maintaining good performance.
    
    Key concepts:
    - Multi-Head Attention: Split attention into multiple "heads" for parallel processing
    - Grouped Query Attention: Share key/value heads across multiple query heads
    - RoPE (Rotary Position Embedding): Adds positional information to queries and keys
    - Query/Key Normalization: Normalizes queries and keys for better training stability
    
    Architecture:
    1. Linear projections: input -> [Q, K, V] matrices
    2. Reshape and transpose for multi-head format
    3. Apply RoPE to Q and K for positional encoding
    4. Repeat K,V for GQA (if num_key_value_heads < num_attention_heads)
    5. Compute attention scores: Q @ K^T / sqrt(head_dim)
    6. Apply causal mask (for autoregressive generation)
    7. Softmax + dropout
    8. Apply attention to values: attention @ V
    9. Reshape and project output
    
    GQA efficiency:
    - Standard MHA: num_heads separate K,V pairs
    - GQA: num_key_value_heads < num_attention_heads, K,V are shared
    - Reduces memory usage and computation while maintaining quality
    """

    def __init__(self, config: RocLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        
        # Calculate the number of key-value groups for GQA
        # Hint: This determines how many query heads share each key-value pair
        # Formula: num_attention_heads / num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        
        # Calculate attention scaling factor
        # Hint: This is 1/sqrt(head_dim) to prevent attention scores from becoming too large
        # Why? Large scores after softmax can cause vanishing gradients
        self.scaling = self.head_dim**-0.5
        
        self.attention_dropout = config.attention_dropout
        self.is_causal = True  # For autoregressive language modeling

        # Initialize linear projections for Q, K, V, and output
        # Hint: 
        # - q_proj: hidden_size -> num_attention_heads * head_dim
        # - k_proj: hidden_size -> num_key_value_heads * head_dim (smaller for GQA)
        # - v_proj: hidden_size -> num_key_value_heads * head_dim (smaller for GQA)
        # - o_proj: num_attention_heads * head_dim -> hidden_size
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Initialize query and key normalization layers
        # Hint: These normalize Q and K before computing attention scores
        # This helps with training stability and can improve performance
        self.q_norm = RocLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RocLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention with GQA
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_embeddings: Tuple of (cos, sin) tensors for RoPE
            attention_mask: Optional attention mask for causal attention
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Store input shape for later reshaping
        input_shape = hidden_states.shape[:-1]  # (batch_size, seq_len)
        hidden_shape = (*input_shape, -1, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)

        # TODO: Project and reshape Q, K, V
        # Hint: 
        # 1. Apply linear projections (q_proj, k_proj, v_proj)
        # 2. Reshape to separate heads: (batch, seq_len, num_heads, head_dim)
        # 3. Transpose to (batch, num_heads, seq_len, head_dim) for attention computation
        # 4. Apply normalization to Q and K (but not V)
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # TODO: Apply Rotary Position Embedding (RoPE)
        # Hint: RoPE adds positional information to queries and keys
        # This helps the model understand the relative positions of tokens
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # TODO: Repeat key-value states for Grouped Query Attention
        # Hint: In GQA, we have fewer K,V heads than Q heads
        # We need to repeat K,V to match the number of Q heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # TODO: Compute attention scores
        # Hint: 
        # 1. Matrix multiply Q and K^T to get attention scores
        # 2. Scale by 1/sqrt(head_dim) to prevent large values
        # 3. Shape: (batch, num_heads, seq_len, seq_len)
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        # TODO: Apply attention mask (causal mask for autoregressive generation)
        # Hint: Causal mask ensures each token can only attend to previous tokens
        # This is crucial for language modeling (can't see the future)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # TODO: Apply softmax and dropout
        # Hint:
        # 1. Softmax converts scores to probabilities (sum to 1)
        # 2. Use float32 for numerical stability, then convert back
        # 3. Apply dropout only during training
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        # TODO: Apply attention to values and reshape
        # Hint:
        # 1. Matrix multiply attention weights with values
        # 2. Transpose back to (batch, seq_len, num_heads, head_dim)
        # 3. Reshape to (batch, seq_len, hidden_size)
        # 4. Apply output projection
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class RocLMDecoderLayer(nn.Module):
    """Decoder layer for RocLM"""

    def __init__(self, config: RocLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = RocLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = RocLMMLP(config)
        self.input_layernorm = RocLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RocLMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class RocLMModel(nn.Module):
    """
    RocLM model implementation - The main transformer decoder
    
    This is the core transformer model that combines all the components:
    - Token embeddings: Convert token IDs to vectors
    - Positional encoding: Add position information using RoPE
    - Multiple decoder layers: Each with self-attention and MLP
    - Final normalization: RMSNorm before output
    
    Architecture flow:
    1. Input tokens -> Embedding layer
    2. Add positional encoding (RoPE)
    3. Pass through N decoder layers (attention + MLP)
    4. Final normalization
    5. Output hidden states for language modeling head
    
    This is a decoder-only model, perfect for autoregressive language generation.
    """

    def __init__(self, config: RocLMConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                RocLMDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RocLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RocLMRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, device=inputs_embeds.device
            ).unsqueeze(0)

        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.tril(
                torch.ones(seq_length, seq_length, device=inputs_embeds.device)
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                inputs_embeds.dtype
            ).min

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Process through all layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class RocLMForCausalLM(nn.Module):
    """RocLM for causal language modeling"""

    def __init__(self, config: RocLMConfig):
        super().__init__()
        self.config = config
        self.model = RocLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified
        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        """Tie the weights of input embeddings and output projection"""
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for causal language modeling"""
        hidden_states = self.model(input_ids, **kwargs)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate text autoregressively"""
        self.eval()
        generated = input_ids.clone()

        max_new_tokens = max_length - input_ids.shape[1]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated)[:, -1, :] / temperature

                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(
                            logits, min(top_k, logits.size(-1))
                        )
                        filtered_logits = torch.full_like(logits, float("-inf"))
                        filtered_logits = filtered_logits.scatter(
                            1, top_k_indices, top_k_logits
                        )
                        logits = filtered_logits

                    # Apply top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            logits, descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            F.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float("-inf")

                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if we hit max length
                if generated.shape[1] >= max_length:
                    break

        return generated


def create_roclm_model(config_dict: dict = None) -> RocLMForCausalLM:
    """Create a RocLM model with specified configuration"""
    if config_dict is None:
        config_dict = {}

    config = RocLMConfig(**config_dict)
    return RocLMForCausalLM(config)


if __name__ == "__main__":
    # Test the RocLM implementation
    print("🧪 Testing RocLM Implementation")
    print("=" * 60)

    # Create model with default config
    model = create_roclm_model()
    print(
        f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")

        # Test generation
        generated = model.generate(input_ids[:1], max_length=20, temperature=0.8)
        print(f"Generated shape: {generated.shape}")

    print("\n✅ RocLM model test completed!")
