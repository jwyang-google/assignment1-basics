from .models import *
from .tokenizer import BPE_Tokenizer
import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np
import os, typing


def load_data(data, batch_size, context_length, device):
  data_len = data.shape[0]
  assert data_len - context_length >= batch_size

  input_tensors = np.zeros((batch_size, context_length))
  target_tensors = np.zeros((batch_size, context_length))
  
  batch_idx = np.random.choice(np.arange(data_len - context_length, dtype=np.int8), size=batch_size, replace=False)
  target_idx = batch_idx + 1
  
  for i in range(batch_size):
    input_tensors[i, :] = data[batch_idx[i]:batch_idx[i]+context_length]
    target_tensors[i, :] = data[target_idx[i]:target_idx[i]+context_length]

  input_tensors = torch.Tensor(input_tensors).to(device)
  target_tensors = torch.Tensor(target_tensors).to(device)

  return input_tensors, target_tensors


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.nn.Module, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
  model_state = model.state_dict()
  optimizer_state = optimizer.state_dict()
  torch.save({
    "iteration": iteration, 
    "model_state": model_state, 
    "optimizer_state": optimizer_state
  }, out)


def load_checkpoint(
    src, model, optimizer=None
):
  ckpt = torch.load(src)
  model.load_state_dict(ckpt['model_state'])
  if optimizer:
    optimizer.load_state_dict(ckpt['optimizer_state'])
  iter = ckpt['iteration']
  return int(iter)


# batch, vocab_size
# batch
def cross_entropy_loss(predicted, target):
  # neg log softmax
  predicted_max, _ = torch.max(predicted, dim=-1, keepdim=True)
  predicted = predicted - predicted_max
  neg_log_softmax = -1 * (predicted - torch.log(torch.sum(torch.exp(predicted), dim=-1, keepdim=True)))

  # index into target
  batch_idx = torch.arange(predicted.size(0))
  loss = neg_log_softmax[batch_idx, target]
  return loss.mean()


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
  if it > cosine_cycle_iters:
    return min_learning_rate

  if it < warmup_iters:
    return max_learning_rate / warmup_iters * it

  if warmup_iters <= it <= cosine_cycle_iters:
    cosine_param = math.cos(
        math.pi / (cosine_cycle_iters - warmup_iters) * (it - warmup_iters)
    )
    return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
        1 + cosine_param
    )
    
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
  params = [p for p in parameters if p.grad is not None]
  if not params:
    return

  total_norm_sq = torch.zeros(1, device=params[0].grad.device)
  for p in params:
    total_norm_sq += p.grad.pow(2).sum()

  total_norm = torch.sqrt(total_norm_sq)

  if total_norm > max_l2_norm:
    scale = max_l2_norm / (total_norm + eps)
    for p in params:
      p.grad.mul_(scale)


class SGD(torch.optim.Optimizer):
  def __init__(self, params, lr=1e-3):
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")
    
    defaults = {"lr": lr}
    super().__init__(params, defaults)
  
  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()
    for group in self.param_groups:
      lr = group["lr"] # Get the learning rate.
      
      for p in group["params"]:
        if p.grad is None:
          continue
        
        state = self.state[p] # Get state associated with p.
        t = state.get("t", 0) # Get iteration number from the state, or initial value.
        grad = p.grad.data # Get the gradient of loss with respect to p.
        p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
        state["t"] = t + 1 # Increment iteration number.
    
    return loss
  

class AdamW(torch.optim.Optimizer):
  def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e10-8):
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")

    defaults = {"lr": lr, "weight_decay": weight_decay, "beta1": betas[0], "beta2": betas[1], "eps": eps}
    super().__init__(params, defaults)
  
  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure

    for group in self.param_groups:
      lr = group["lr"]
      weight_decay = group["weight_decay"]
      beta1 = group["beta1"]
      beta2 = group["beta2"]
      eps = group["eps"]

      for param in group["params"]:
        grad = param.grad
        if grad is None:
          continue
        
        # get current states
        state = self.state[param]
        t = state.get('t', 1)
        m = state.get('m', torch.zeros_like(param))
        v = state.get('v', torch.zeros_like(param))

        # update params
        grad = grad.data
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        adjusted_lr = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
        param.data = param.data - adjusted_lr * m / (v ** 0.5 + eps)
        param.data = param.data - lr * weight_decay * param.data

        # set states
        state['t'] = t + 1
        state['m'] = m
        state['v'] = v


class Train():

  def __init__(
      self, 
      # device for training
      device: str,

      # tokenizer
      training_filepath: str | os.PathLike,
      validation_filepath: str | os.PathLike,
      vocab_filepath: str | os.PathLike,
      merges_filepath: str | os.PathLike,
      special_tokens: list[str] | None,

      # training loop hyperparams
      batch_size: int, 
      
      # model params
      vocab_size: int,
      context_length: int,
      num_layers: int,
      d_model: int,
      n_heads: int,
      d_ff: int,
      dtype: torch.dtype | None, 

      # optimizer params
      lr: float=1e-3, 
      weight_decay: float=1e-2, 
      betas: tuple[float, float]=(0.9, 0.999),
      eps: float=1e10-8,
    
      # architecture hyperparams
      apply_rope: bool = True,
      rope_theta: float | None = None,
      apply_nope: bool = False,
      pre_norm: bool = True,
      post_norm: bool = False,
      ffn_type: str = "swiglu"
  ):
    self.device = device
    self.dtype = dtype

    self.batch_size = batch_size
    self.training_filepath = training_filepath
    self.validation_filepath = validation_filepath

    self.context_length = context_length

    # init tokenizer
    self.tokenizer = BPE_Tokenizer(
      train_from_scratch=False, 
      vocab_filepath=vocab_filepath,
      merges_filepath=merges_filepath,
      vocab_size=vocab_size,
      special_tokens=["<|endoftext|>"]
    )
    
    # init model
    self.model = TransformerLM(
      vocab_size, 
      context_length, 
      num_layers, 
      d_model, 
      n_heads, 
      d_ff, 
      apply_rope, 
      theta=rope_theta,
      device=device
    )

    # init optimizer
    self.optimizer = AdamW(
      self.model.state_dict(),
      lr=lr,
      weight_decay=weight_decay,
      betas=betas,
      eps=eps
    )
  
  def train(self, num_iters: int):
    # load data
    training_data = np.memmap(self.corpus, dtype="uint16", mode="r")
    total_positions = training_data.shape[0] - self.context_length - self.batch_size

    # split data into multiple chunks
    for i in range(num_iters):
      # get a random position
      position = np.random.choice(np.arange(total_positions, dtype=np.int32))

      # get batch
      batched_data, batched_targets = load_data(
        training_data[position:position+self.batch_size+self.context_length], 
        self.batch_size, 
        self.context_length, 
        device=self.device
      )

      logits = self.model.forward(batched_data)
      loss = cross_entropy_loss(logits, batched_targets)
      print("iteration: {}, loss: {}".format(i, loss))

      loss.backward()
      self.optimizer.step()    

      # run on validation dataset & save checkpoint
      if i % 1000 == 0:
        # TODO - save checkpoint
        pass
        # TODO - run on validation dataset


def training_loop_example(learning_rate=1):
  weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
  opt = SGD([weights], lr=learning_rate)
  for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print("iteration {}, loss: {}".format(t, loss.cpu().item()))
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.

# training_loop_example(learning_rate=1e1)
# training_loop_example(learning_rate=1e2)
# training_loop_example(learning_rate=1e3)