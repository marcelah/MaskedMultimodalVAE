import jax.numpy as jnp
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
from flax.linen.normalization import _normalize, _compute_stats
import neural_networks
from neural_networks import MLP
from flax.linen.module import Module
from typing import (Any, Callable, Iterable, Optional, Tuple, Union)
PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
tfd = tfp.distributions
Axes = Union[int, Any]


class SelfAttentionPooling(nn.Module):
  input_dim: int
  output_dim: int
  qkv_dim: int
  mlp_dim: int
  pooling_dim: int
  pooling_depth: int
  act_fn: callable
  num_heads: int = 8
  time_aggregation: bool = False
  num_mixtures: int = 0
  pre_norm: bool = True
  num_attention_layers: int = 2
  use_cross_attention: bool = False

  @nn.compact
  def __call__(self, x, mask):
    if not self.time_aggregation:
      T = 1
      # apply multi-head attention block at each time step over the different modalities
      x_time_batch = jnp.transpose(x, [1, 0, 2])
      x_mask_time_batch = jnp.transpose(mask, [1, 0])
      h = x_time_batch

      g = nn.Dense(features=self.mlp_dim)(h)
      g = self.act_fn(g)
      g = nn.Dense(features=self.pooling_dim)(g)

      #masking
      #g = jnp.where(x_mask_time_batch[:, :, None], g, 0.)

      equivariant_features = g
      self_attention_weight = []
      if self.pre_norm:
        for l in range(self.num_attention_layers):
          attention_block = neural_networks.PreLNMultiHeadAttentionBlock(
            num_heads=self.num_heads,
            qkv_dim=self.qkv_dim,
            rFF_hidden_size=self.mlp_dim,
            out_dim=self.pooling_dim)

          equivariant_features = attention_block(
            equivariant_features,
            equivariant_features,
            jnp.einsum('...t,...s->...ts', x_mask_time_batch, x_mask_time_batch))

        equivariant_features = nn.LayerNorm()(equivariant_features)

      else:
        for l in range(self.num_attention_layers):
          attention_block = neural_networks.MultiHeadAttentionBlock(
            num_heads=self.num_heads,
            qkv_dim=self.qkv_dim,
            rFF_hidden_size=self.mlp_dim,
            out_dim=self.pooling_dim)
          equivariant_features = attention_block(
            equivariant_features,
            equivariant_features,
            jnp.einsum('...t,...s->...ts', x_mask_time_batch, x_mask_time_batch))

      aggregated_features = jnp.sum(jnp.where(x_mask_time_batch[:, :, None], equivariant_features, 0), -2)

      if self.num_mixtures == 0:
        h = MLP(output_dim_feature=self.output_dim,
                hidden_dim_feature=self.mlp_dim,
                hidden_layers=self.pooling_depth,
                act_fn=self.act_fn,
                masked=False)(aggregated_features, None)
        aggregation_logs = {
                            ('equivariant_features', 'vector'): equivariant_features,
                            ('uni-modal features', 'vector'): x,
                            ('projected uni-modal featuers', 'vector'): g,
                            ('aggregated_features', 'vector'): aggregated_features,
                            ('h', 'vector'):  h}
        return h, aggregation_logs
      else:
        h = MLP(output_dim_feature=self.output_dim,
                hidden_dim_feature=self.mlp_dim,
                hidden_layers=self.pooling_depth,
                act_fn=self.act_fn,
                masked=False)(aggregated_features, None)
        aggregation_logs = {
                            ('equivariant_features', 'vector'): equivariant_features,
                            ('uni-modal features', 'vector'): x,
                            ('projected uni-modal featuers', 'vector'): g,
                            ('aggregated_features', 'vector'): aggregated_features,
                            ('h', 'vector'):  h}
        if self.use_cross_attention:
          return (h[:, 0, :-self.num_mixtures], h[:, 0, -self.num_mixtures:]), aggregation_logs
        else:
          return (h[:, :-self.num_mixtures], h[:, -self.num_mixtures:]), aggregation_logs


class PoE(nn.Module):
  output_dim: int
  min_var: float = 1e-3
  @nn.compact
  def __call__(self, x, mask):
    mu, sigma = jnp.split(x, 2, axis=-1)
    var = self.min_var + jnp.exp(2*sigma)
    poe_mean = jnp.sum(jnp.where(jnp.expand_dims(mask, -1),mu*var, 0), 0)/\
               jnp.sum(jnp.where(jnp.expand_dims(mask, -1),1/var, 0), 0)
    poe_var = 1/jnp.sum(jnp.where(jnp.expand_dims(mask, -1), 1/var, 0), 0)
    aggregation_logs = {('uni-modal features', 'vector'): x,
                        ('poe_mean', 'vector'): poe_mean,
                        ('poe_var', 'vector'): poe_mean}
    return jnp.concatenate([poe_mean, .5*jnp.log(poe_var)], axis=-1), aggregation_logs


class MoE(nn.Module):
  output_dim: int
  @nn.compact
  def __call__(self, x, mask):
    aggregation_logs = {('uni-modal features', 'vector'): x}
    return x, aggregation_logs


class SumPooling(nn.Module):
  output_dim: int
  pooling_size: int
  hidden_layer_size: int
  hidden_layer_size_pooling: int
  act_fn: callable = nn.relu
  time_aggregation: bool = False
  hidden_layers_pooling: int = 1
  hidden_layers: int = 2
  num_mixtures: int = 0
  residual: bool = False

  @nn.compact
  def __call__(self, x, mask):
    if not self.residual:
      g = MLP(output_dim_feature=self.pooling_size,
            hidden_dim_feature=self.hidden_layer_size,
            hidden_layers=self.hidden_layers,
            act_fn=self.act_fn
            )(
        x, mask[:,:,None])
    else:

      g = jnp.where(mask[:, :, None], x, 0)
      for _ in range(self.hidden_layers - 1):
        f = nn.Dense(features=g.shape[-1], use_bias=False)(g)
        f_norm = SetNorm()(f)
        f = self.act_fn(f_norm)
        f = nn.Dense(features=f.shape[-1], use_bias=False)(f)
        f_norm = SetNorm()(f)
        f_norm = jnp.where(mask[:, :, None], f_norm, 0)
        g = g + f_norm
        g = jnp.where(mask[:, :, None], g, 0)
      g = nn.Dense(features=self.pooling_size)(g)
      g = jnp.where(mask[:, :, None], g, 0)

    modality_sum = jnp.sum(jnp.where(mask[:, :, None], g, 0), 0)

    if self.time_aggregation:
      pass
    else:
      if not self.residual:
        invariant_aggregation = MLP(output_dim_feature=self.output_dim,
                                  hidden_dim_feature=self.hidden_layer_size_pooling,
                                  hidden_layers=self.hidden_layers_pooling,
                                  act_fn=self.act_fn,
                                  masked=False)(
        modality_sum, None)
        h = invariant_aggregation

      else:
        h = nn.Dense(features=self.hidden_layer_size_pooling, use_bias=False)(modality_sum)
        for _ in range(self.hidden_layers_pooling - 1):
          f = nn.LayerNorm()(h)
          f = self.act_fn(f)
          f = nn.Dense(self.hidden_layer_size_pooling, use_bias=False)(f)
          f = nn.LayerNorm()(f)
          f = self.act_fn(f)
          f = nn.Dense(features=self.hidden_layer_size_pooling, use_bias=False)(f)
          h = h + f
        h = nn.LayerNorm()(h)
        h = nn.Dense(self.output_dim)(h)


    aggregation_logs = {('uni-modal features', 'vector'): x,
                        ('projected uni-modal featuers', 'vector'): g,
                        ('aggregated_features', 'vector'): modality_sum,
                        ('h', 'vector'): h}

    if self.num_mixtures == 0:
      return h, aggregation_logs
    else:
      return (h[:, :-self.num_mixtures], h[:, -self.num_mixtures:]), aggregation_logs


class SumPoolingEquivariant(nn.Module):
  output_dim: int
  pooling_size: int
  hidden_layer_size: int
  hidden_layer_size_pooling: int
  act_fn: callable = nn.relu
  time_aggregation: bool = False
  hidden_layers_pooling: int = 1
  hidden_layers: int = 2
  num_mixtures: int = 0
  residual: bool = False

  @nn.compact
  def __call__(self, x, z, mask):
    if not self.residual:
      g_pool = MLP(output_dim_feature=self.pooling_size,
            hidden_dim_feature=self.hidden_layer_size,
            hidden_layers=self.hidden_layers,
            act_fn=self.act_fn
            )(
        x, mask[:,:,None])
      g_equiv = MLP(output_dim_feature=self.pooling_size,
            hidden_dim_feature=self.hidden_layer_size,
            hidden_layers=self.hidden_layers,
            act_fn=self.act_fn
            )(
        x, mask[:,:,None])
      g_z = MLP(output_dim_feature=self.pooling_size,
            hidden_dim_feature=self.hidden_layer_size,
            hidden_layers=self.hidden_layers,
            act_fn=self.act_fn
            )(
        z, mask[:,:,None])
    else:
      pass


    modality_sum = jnp.sum(jnp.where(mask[:, :, None], g_pool , 0), 0)
    equivariant_terms = modality_sum + g_z + g_equiv

    if self.time_aggregation:
      pass
    else:
      equivariant_result = MLP(output_dim_feature=self.output_dim,
                                  hidden_dim_feature=self.hidden_layer_size_pooling,
                                  hidden_layers=self.hidden_layers_pooling,
                                  act_fn=self.act_fn,
                                  masked=False)(
        equivariant_terms, None)

    aggregation_logs = {('uni-modal features', 'vector'): x,
                        ('aggregated_features', 'vector'): modality_sum,
                        ('equivariant_result', 'vector'): equivariant_result}

    if self.num_mixtures == 0:
      return equivariant_result, aggregation_logs
    else:
      pass
      #not implemented


class SelfAttentionEquivariant(nn.Module):
  input_dim: int
  output_dim: int
  qkv_dim: int
  mlp_dim: int
  pooling_dim: int
  pooling_depth: int
  act_fn: callable
  num_heads: int = 8
  time_aggregation: bool = False
  num_mixtures: int = 0
  pre_norm: bool = True
  num_attention_layers: int = 2
  use_cross_attention: bool = False

  @nn.compact
  def __call__(self, x, z, mask):
    if not self.time_aggregation:
      T = 1
      # apply multi-head attention block at each time step over the different modalities
      x_time_batch = jnp.transpose(x, [1, 0, 2])
      x_mask_time_batch = jnp.transpose(mask, [1, 0])
      h = x_time_batch

      g = nn.Dense(features=self.mlp_dim)(h)
      g = self.act_fn(g)
      g = nn.Dense(features=self.pooling_dim)(g)

      g_z = nn.Dense(features=self.mlp_dim)(z)
      g_z = self.act_fn(g_z)
      g_z = nn.Dense(features=self.pooling_dim)(g_z)

      #masking
      #g = jnp.where(x_mask_time_batch[:, :, None], g, 0.)

      equivariant_features = g + g_z
      self_attention_weight = []
      if self.pre_norm:
        for l in range(self.num_attention_layers):
          attention_block = neural_networks.PreLNMultiHeadAttentionBlock(
            num_heads=self.num_heads,
            qkv_dim=self.qkv_dim,
            rFF_hidden_size=self.mlp_dim,
            out_dim=self.pooling_dim)

          equivariant_features, = attention_block(
            equivariant_features,
            equivariant_features,
            jnp.einsum('...t,...s->...ts', x_mask_time_batch, x_mask_time_batch))


      equivariant_features = nn.LayerNorm()(equivariant_features)
      equivariant_features = nn.Dense(features=self.output_dim)(equivariant_features)

      aggregation_logs = {('equivariant_features', 'vector'): equivariant_features,
                            ('uni-modal features', 'vector'): x,
                            ('projected uni-modal featuers', 'vector'): g,
                            ('h', 'vector'):  h}
      return jnp.transpose(equivariant_features, [1, 0 ,2]), aggregation_logs


class SetNorm(Module):
  epsilon: float = 1e-6
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones

  @nn.compact
  def __call__(self, x):
    mean, var = _compute_stats(x, [-2, -1], self.dtype)
    return _normalize(
        self, x, mean, var, [-2, -1], [-1],
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init)

