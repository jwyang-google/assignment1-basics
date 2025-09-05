import torch
from .models import *
from .train import *
import os


class Decoder():
  def __init__(
    self,
    # model checkpoint
    checkpoint_path: str | os.PathLike, 

    # TODO: tokenizer params if needed

    # model params
    vocab_size: int, 
    context_length: int,
    num_layers: int, 
    d_model: int, 
    n_heads: int, 
    d_ff: int, 
    apply_rope: bool = False,
    theta: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,

    # eos_token_id
    eos: int | None = None,

    # sampling params
    sampling_method: str = "top_p",
    temperature: float = 1.0
  ):
    # init model
    self.model = TransformerLM(
      vocab_size,
      context_length,
      num_layers,
      d_model,
      n_heads,
      d_ff,
      apply_rope,
      theta,
      device,
      dtype
    )

    # load model weights
    load_checkpoint(checkpoint_path, self.model, optimizer=None)
    self.model.load_state_dict()

    # eos_token_id
    self.eos_token_id = eos

    # sampling params
    self.sampling_method = sampling_method
    self.temperature = temperature

  def decode(self, prompt_ids, max_tokens):
    sampling_method = None
    if self.sampling_method == "top_p":
      sampling_method = self.top_p_sampling
    elif self.sampling_method == "top_k":
      sampling_method = self.top_k_sampling
    elif self.sampling_method == "beam":
      sampling_method = self.beam
    else:
      raise NotImplementedError

    for _ in range(max_tokens):
      logits = self.model.forward(prompt_ids)
      next_id = sampling_method(logits)
      prompt_ids.append(next_id)
      if next_id == self.eos_token_id:
        break
    
    return prompt_ids


  # sample one token from logits
  def top_p_sampling(
      self,
      logits: torch.Tensor,
  ):
    # apply temperature scaling
    raise NotImplementedError
  

  def top_k_sampling(
      self,
      logits: torch.Tensor
  ):
    raise NotImplementedError
  
  def beam(
      self,
      logits: torch.Tensor
  ):
   raise NotImplementedError