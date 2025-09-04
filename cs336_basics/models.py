import torch
from einops import rearrange, einsum


class Linear(torch.nn.Module):

  def __init__(
      self, 
      in_features: int,
      out_features: int,
      device: torch.device | None = None,
      dtype: torch.dtype | None = None
  ):
    # call superclass instructor
    super().__init__()

    # initialize weights
    init_mean = 0.0
    init_std = (2.0/(in_features + out_features)) ** 0.5
    self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(
      torch.empty((out_features, in_features), device=device, dtype=dtype), 
      mean=init_mean, 
      std=init_std,
      a=-3.0 * init_std,
      b=3.0 * init_std))
    
  def forward(
      self,
      x: torch.Tensor
  ):
    res = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    return res


class Embedding(torch.nn.Module):

  def __init__(
      self,
      vocab_size: int,
      d_model: int,
      device: torch.device | None = None,
      dtype: torch.dtype | None = None
  ):
    super().__init__()

    # initialize embedding weights
    self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(
      torch.empty((vocab_size, d_model), device=device, dtype=dtype),
      mean=0.0,
      std=1.0,
      a=-3.0,
      b=3.0
    ))

  def forward(self, x: torch.Tensor):
    res = self.weight[x]
    return res
  

class RMSNorm(torch.nn.Module):

  def __init__(
      self,
      d_model: int,
      eps: float = 1e-5,
      device: torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.d_model = d_model
    self.eps = eps

    # initialize weights
    self.weight = torch.nn.Parameter(torch.ones((d_model, ), dtype=dtype, device=device))

  def forward(self, x: torch.Tensor):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    rms = ((torch.sum(x ** 2, axis=2)/self.d_model) + self.eps) ** 0.5
    x = x * self.weight / torch.unsqueeze(rms, 2)
    return x.to(in_dtype)
  

class MLP(torch.nn.Module):

  def __init__(
      self,
      d_model: int,
      d_ff: int,
      device: torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()

    # define layers
    self.w1 = Linear(d_model, d_ff, device, dtype)
    self.w3 = Linear(d_model, d_ff, device, dtype)
    self.w2 = Linear(d_ff, d_model, device, dtype)


  def forward(self, x: torch.Tensor):
    x1 = self.w1.forward(x)
    x3 = self.w3.forward(x)

    x1 = torch.sigmoid(x1) * x1
    x = x1 * x3
    x = self.w2.forward(x)
    return x
  

class RoPE(torch.nn.Module):

  def __init__(
      self, 
      d_k: int,
      theta: float,
      max_seq_len: int,
  ):
    super().__init__()

    self.d_k = d_k
    self.max_seq_len = max_seq_len
    
    # Construct theta
    half_dim = d_k // 2
    freqs = torch.arange(half_dim, dtype=torch.float32)
    inv_freqs = 1.0 / (theta ** ((freqs * 2.0)/d_k))

    seq_idx = torch.arange(max_seq_len, dtype=torch.float32)
    theta_matrix = einsum(seq_idx, inv_freqs, "i, j -> i j")

    self.cosine_theta = torch.cos(theta_matrix)
    self.sin_theta = torch.sin(theta_matrix)

    self.register_buffer("cost", self.cosine_theta, persistent=False)
    self.register_buffer("sint", self.sin_theta, persistent=False)

  # in_features: (batch, sequence, d_k)
  # token_positions: (batch, sequence)
  # sin/cosine_theta: (max_seq_len, d_k//2)
  def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor):
    in_dtype = in_features.dtype
    in_features = in_features.to(torch.float32)

    in_features = rearrange(in_features, "... (half_d_k two) -> ... half_d_k two", two=2)
    first_in_features, second_in_features = in_features.unbind(dim=-1)

    cos_theta = self.cost[token_positions]
    sin_theta = self.sint[token_positions]

    cos_theta = torch.unsqueeze(cos_theta, 0)
    sin_theta = torch.unsqueeze(sin_theta, 0)

    first_half = cos_theta * first_in_features - sin_theta * second_in_features
    second_half = cos_theta * second_in_features + sin_theta * first_in_features

    res = torch.stack((first_half, second_half), dim=-1)
    res = rearrange(res, "... half_d_k two -> ... (half_d_k two)", two=2)
    return res.to(in_dtype)
  

def softmax(x: torch.Tensor, dim: int):
  in_dtype = x.dtype
  x = x.to(torch.float32)

  max_elem, _ = torch.max(x, dim=dim, keepdim=True)
  res = torch.exp(x - max_elem)
  res = res / torch.sum(res, dim=dim, keepdim=True)
  return res.to(in_dtype)


def scaled_dot_product_attention(
    query: torch.Tensor, # (batch, query_seq, d)
    key: torch.Tensor, # (batch, key_seq, d)
    value: torch.Tensor, # (batch, value_seq, d)
    mask: torch.Tensor
):
  d_k = query.shape[-1]
  qk_product = einsum(query, key, "... query_seq d, ... key_seq d -> ... query_seq key_seq")
  qk_product = qk_product / (d_k ** 0.5)

  # apply mask to qk_product
  qk_product = torch.where(mask, qk_product, float('-inf'))
  w = softmax(qk_product, dim=-1)

  # compute wv product
  wv_product = einsum(w, value, "... query_seq key_seq, ... key_seq d -> ... query_seq d")
  return wv_product


class MultiHeadSelfAttention(torch.nn.Module):

  def __init__(
      self, 
      d_model: int, 
      n_heads: int,
      apply_rope: bool = False,
      theta: float | None = None,
      max_seq_len: int | None = None,
      device: torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()

    self.d_model = d_model
    self.n_heads = n_heads
    self.head_dim = self.d_model // self.n_heads

    # Q, K, V Projection Matrix
    # self.proj_layer = Linear(d_model, 3*d_model, device, dtype)
    self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)


    # rope layer
    self.apply_rope = apply_rope
    if apply_rope:
      assert theta is not None
      assert max_seq_len is not None
      self.rope = RoPE(self.head_dim, theta, max_seq_len)
      
    # out projection layer
    self.output_proj = Linear(d_model, d_model, device, dtype)

  def forward(self, x: torch.Tensor, token_posiitons: torch.Tensor=None):
    # QKV projection
    # qkv_projection = self.proj_layer(x)
    # qkv_projection = rearrange(qkv_projection, "... (three d_k) -> ... three d_k", three=3)
    # q = qkv_projection[:, :, 0, :]
    # k = qkv_projection[:, :, 1, :]
    # v = qkv_projection[:, :, 2, :]
    # q, k, v = qkv_projection.chunk(3, dim=-1)
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # self attetion with causal mask
    q = rearrange(q, "... seq (n_heads head_dim) -> ... n_heads seq head_dim", n_heads=self.n_heads)
    k = rearrange(k, "... seq (n_heads head_dim) -> ... n_heads seq head_dim", n_heads=self.n_heads)
    v = rearrange(v, "... seq (n_heads head_dim) -> ... n_heads seq head_dim", n_heads=self.n_heads)

    if self.apply_rope:
      q = self.rope.forward(q, token_posiitons)
      k = self.rope.forward(k, token_posiitons)

    seq_len = x.shape[-2]
    rows = torch.arange(seq_len).unsqueeze(dim=1)
    cols = torch.arange(seq_len).unsqueeze(dim=0)
    mask = rows >= cols
    attention = scaled_dot_product_attention(q, k, v, mask)
    
    attention = rearrange(attention, "... n_heads seq head_dim -> ... seq (n_heads head_dim)", n_heads=self.n_heads)
    output = self.output_proj(attention)
    return output


class TransformerBlock(torch.nn.Module):
  def __init__(
      self,
      d_model: int,
      n_heads: int,
      d_ff: int,
      apply_rope: bool = False,
      theta: float | None = None,
      max_seq_len: int | None = None,
      device: torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    
    # pre normalization layer
    self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)

    # attention layer
    self.attn = MultiHeadSelfAttention(
      d_model,
      n_heads,
      apply_rope=apply_rope,
      theta=theta,
      max_seq_len=max_seq_len,
      device=device,
      dtype=dtype
    )

    # MLP layer
    self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
    self.ffn = MLP(d_model, d_ff, device, dtype)

  
  def forward(self, x: torch.Tensor, token_positions=None):
    if token_positions is None:
        token_positions = torch.arange(x.size(1), device=x.device).expand(
            x.size(0), -1
        )

    orig_x = x
    x = self.ln1(x)
    x = self.attn(x, token_positions)
    x = orig_x + x

    orig_x = x
    x = self.ln2(x)
    x = self.ffn(x)
    x = orig_x + x
    return x


class TransformerLM(torch.nn.Module):

  def __init__(
      self, 
      vocab_size: int, 
      context_length: int,
      num_layers: int, 
      d_model: int, 
      n_heads: int, 
      d_ff: int, 
      apply_rope: bool = False,
      theta: float | None = None,
  ):
    super().__init__()
    # embedding layer
    self.token_embeddings = Embedding(vocab_size, d_model)
    
    # list of transformer blocks
    self.layers = torch.nn.ModuleList(
      [TransformerBlock(d_model, n_heads, d_ff, apply_rope, theta, context_length) for _ in range(num_layers)]
    )

    # output norm
    self.ln_final = RMSNorm(d_model=d_model)

    # output linear projection
    self.lm_head = Linear(d_model, vocab_size)

  
  def forward(self, x: torch.Tensor, token_positions=None):
    x = self.token_embeddings(x)

    if token_positions is None:
      token_positions = torch.arange(x.size(1), device=x.device).expand(
          x.size(0), -1
    )

    for layer in self.layers:
      x = layer(x, token_positions)

    x = self.ln_final(x)
    x = self.lm_head(x)
    return x


