import jax.numpy as jnp
from flax import linen as nn
from typing import (Any, Callable, Optional, Tuple)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

class MLP(nn.Module):
  output_dim_feature: int
  hidden_dim_feature: int
  hidden_layers: int = 1
  masked: bool = True
  kernel_init: bool = None
  act_fn: callable = nn.relu

  @nn.compact
  def __call__(self, x, mask):
    if self.masked:
      x = jnp.where(mask, x, 0.)
    z = x
    for i in range(self.hidden_layers):
      if self.kernel_init is None:
        z = nn.Dense(self.hidden_dim_feature, name='fc'+ str(i))(z)
      else:
        z = nn.Dense(self.hidden_dim_feature, name='fc'+ str(i), kernel_init=self.kernel_init)(z)
      z = self.act_fn(z)
    if self.kernel_init is None:
      z = nn.Dense(self.output_dim_feature, name='fc'+str(self.hidden_layers))(z)
    else:
      z = nn.Dense(self.output_dim_feature, name='fc'+str(self.hidden_layers), kernel_init=self.kernel_init)(z)

    if self.masked:
      return jnp.where(mask, z, 0.)
    else:
      return z


class MultiHeadAttentionBlock(nn.Module):
  num_heads: int
  qkv_dim: int
  rFF_hidden_size: int
  out_dim : int

  @nn.compact
  def __call__(self, inputs_q, inputs_kv, mask):
    inputs_q_masked = jnp.where(jnp.expand_dims(mask[:,0,:], -1), inputs_q, 0.)
    h = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads, qkv_features=self.qkv_dim, out_features=self.out_dim, use_bias=False
    )(inputs_q, inputs_kv, mask)
    #need to mask as attention weights are 0/0 ~ 1/M for masked queries
    h = jnp.where(jnp.expand_dims(mask[:,0,:], -1), h, 0.)
    h = inputs_q_masked + h
    z = nn.LayerNorm()(h)
    y = nn.Dense(features=self.rFF_hidden_size)(z)
    y = nn.relu(y)
    y = nn.Dense(features=self.out_dim)(y)
    x = z + y
    x = nn.LayerNorm()(x)
    return x

class PreLNMultiHeadAttentionBlock(nn.Module):
  num_heads: int
  qkv_dim: int
  rFF_hidden_size: int
  out_dim : int

  @nn.compact
  def __call__(self, inputs_q, inputs_kv, mask):
    inputs_q_norm = nn.LayerNorm()(inputs_q)
    inputs_kv_norm = nn.LayerNorm()(inputs_kv)
    h = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads, qkv_features=self.qkv_dim, out_features=self.out_dim, use_bias=False
    )(inputs_q_norm, inputs_kv_norm, mask)
    h = inputs_q + h
    h_norm = nn.LayerNorm()(h)
    y = nn.Dense(features=self.rFF_hidden_size)(h_norm)
    y = nn.relu(y)
    y = nn.Dense(features=self.out_dim)(y)
    x = h + y
    return x


