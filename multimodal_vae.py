import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
import encoding_models
tfd = tfp.distributions
from flax.metrics import tensorboard
import numpy as np
import os
import datetime
from typing import (Callable, Optional)
from flax.training import train_state, checkpoints

class MixturePriorEmbedding(nn.Module):
  K: int
  d: int

  @nn.compact
  def __call__(self, c):
    #assert(self.K <= self.d)
    x = nn.Embed(num_embeddings=self.K, features=2 * self.d)(c)
    return jnp.split(x, 2, -1)


class FixedPriorEmbedding(nn.Module):
  d: int

  @nn.compact
  def __call__(self, c):
    x = jnp.zeros(list(c.shape) + [2 * self.d])
    return jnp.split(x, 2, -1)


def softplus(x):
    return jnp.log(1. + jnp.exp(x))

class MultiModalVAE(nn.Module):
  bound: str
  K: int
  d: int
  x_dims: dict
  x_types: dict
  encoder_models: dict
  decoder_models: dict
  T: int
  pooling_size: int
  hidden_aggregation_size: int
  attention_size: int
  beta: int
  embedding_size: int
  sum_pooling_depth: int = 2
  attention_depth: int = 2
  pre_layer_norm: bool = True
  pooling_aggregation_depth: int = 2
  num_heads: int = 4
  fixed_prior: bool = False
  prior_logits_init: Callable = nn.initializers.constant(1.)
  aggregation: str = 'SumPooling'
  use_data_loader: bool = False
  collate_fn: Optional[bool] = None
  include_prior: bool = True
  prior: str = 'Gaussian'
  posterior: str = 'Gaussian'
  min_encoder_scale: float = 5e-4
  min_decoder_scale: float = 5e-4
  act_fn : callable = nn.relu
  rec_weights: Optional[bool] = None
  stl_grad: Optional[bool] = True
  residual_encoder: Optional[bool] = False

  def setup(self):

    self.prior_logits = self.param('prior_logits', self.prior_logits_init, (self.K,))
    self.decoders = self.decoder_models
    self.marginal_features = self.encoder_models


    if self.fixed_prior:
      assert(1 == self.K)
      self.prior_embedding = nn.vmap(FixedPriorEmbedding,
                                     in_axes=0, out_axes=0,
                                     variable_axes={'params': None},
                                     split_rngs={'params': False})(d=self.d)
    else:
      self.prior_embedding = nn.vmap(MixturePriorEmbedding,
                                   in_axes=0, out_axes=0,
                                   variable_axes={'params': None},
                                   split_rngs={'params': False})(K=self.K, d=self.d)
    if self.aggregation == 'SumPooling':
      self.multimodal_aggregation = nn.vmap(encoding_models.SumPooling, in_axes=0, out_axes=0,
                         variable_axes={'params': None},
                         split_rngs={'params': False})\
          (output_dim=2 * self.d, hidden_layer_size=self.hidden_aggregation_size,
           pooling_size=self.pooling_size, time_aggregation = (self.T>1),
           hidden_layers_pooling=self.pooling_aggregation_depth,
           hidden_layers=self.sum_pooling_depth, hidden_layer_size_pooling=self.hidden_aggregation_size,
           act_fn=self.act_fn, residual=self.residual_encoder)
    elif self.aggregation == 'SumPoolingMixture':
      self.multimodal_aggregation = nn.vmap(encoding_models.SumPooling, in_axes=0, out_axes=0,
                                            variable_axes={'params': None},
                                            split_rngs={'params': False}) \
        (output_dim=(1 + 2 * self.d) * self.K, hidden_layer_size=self.hidden_aggregation_size,
          pooling_size=self.pooling_size, time_aggregation = (self.T>1),
          hidden_layers_pooling=self.pooling_aggregation_depth,
          hidden_layers=self.sum_pooling_depth, hidden_layer_size_pooling=self.hidden_aggregation_size,
          num_mixtures=self.K, act_fn=self.act_fn, residual=self.residual_encoder)
    elif self.aggregation == 'SelfAttention':
      num_heads = self.num_heads
      qkv_dim = (self.attention_size // num_heads) * num_heads #embedding size
      self.multimodal_aggregation = nn.vmap(encoding_models.SelfAttentionPooling, in_axes=0, out_axes=0,
                                            variable_axes={'params': None},
                                            split_rngs={'params': False}) \
        (output_dim=2*self.d, qkv_dim=qkv_dim, time_aggregation = (self.T>1), num_heads=num_heads,
         input_dim=self.embedding_size, mlp_dim=self.hidden_aggregation_size, pooling_dim=self.pooling_size,
         pooling_depth=self.pooling_aggregation_depth, num_attention_layers=self.attention_depth,
         pre_norm=self.pre_layer_norm, act_fn=self.act_fn)
    elif self.aggregation == 'SelfAttentionMixture':
      num_heads = self.num_heads
      qkv_dim = (self.attention_size // num_heads) * num_heads #embedding size
      self.multimodal_aggregation = nn.vmap(encoding_models.SelfAttentionPooling, in_axes=0, out_axes=0,
                                            variable_axes={'params': None},
                                            split_rngs={'params': False}) \
        (output_dim=(1 + 2 * self.d) * self.K, qkv_dim=qkv_dim, time_aggregation = (self.T>1), num_heads=num_heads,
         input_dim=self.embedding_size, mlp_dim=self.hidden_aggregation_size, pooling_dim=self.pooling_size,
         pooling_depth=self.pooling_aggregation_depth, num_attention_layers=self.attention_depth,
         pre_norm=self.pre_layer_norm, num_mixtures=self.K, act_fn=self.act_fn)
    elif self.aggregation == 'PoE':
      assert(self.K == 1)
      self.multimodal_aggregation = nn.vmap(encoding_models.PoE, in_axes=0, out_axes=0,
                                            variable_axes={'params': None},
                                            split_rngs={'params': False}) \
        (output_dim=2 * self.d)
    elif self.aggregation == 'MoE':
      self.multimodal_aggregation = nn.vmap(encoding_models.MoE, in_axes=0, out_axes=0,
                                            variable_axes={'params': None},
                                            split_rngs={'params': False}) \
        (output_dim=2 * self.d)

  def __call__(self, x, x_masks, num_importance_samples, rng, mask_prob=.5, use_rec_weights=False):

    marginal_features = []
    marginal_masks = []
    for k, v in x.items():
      marginal_features.append(jax.vmap(self.marginal_features[k], in_axes=0, out_axes=0)(v, x_masks[k]))
      marginal_masks.append(x_masks[k])

    marginal_features = jnp.stack(marginal_features, 1)
    full_masks = jnp.stack(marginal_masks, 1)
    rng, key = random.split(rng)
    if mask_prob in [0., 1.]:
      sampled_masks = tfd.Bernoulli(probs=(1-mask_prob)*jnp.ones_like(full_masks)
                                  ).sample(seed=key)*full_masks
    else:
      beta_prob = tfd.Uniform(low=0.*jnp.ones_like(full_masks), high=1.*jnp.ones_like(full_masks)
                              ).sample(seed=key)
      rng, key = random.split(rng)
      sampled_masks = tfd.Bernoulli(probs=(1 - beta_prob)
                                    ).sample(seed=key) * full_masks

    complement_masks = full_masks * (1 - sampled_masks)
    if self.aggregation == 'MoE' or (self.aggregation == 'PoE' and self.K == 1):
      #prior features
      marginal_features = jnp.concatenate([marginal_features,
        jnp.concatenate([jnp.tile(jnp.expand_dims(y, -2), [v.shape[0],1,1,1])
                         for y in self.prior_embedding(jnp.expand_dims(jnp.arange(self.K), 0))], -1)], 1)
      if self.aggregation == 'MoE':
        #mask prior features if there is any non-masked modality
        sampled_masks = jnp.where((jnp.sum(sampled_masks, 1) == 0)[:, None],
                jnp.tile(
                  jnp.concatenate([jnp.zeros([len(self.decoders)]), jnp.ones_like(self.prior_logits)])[None, :, None],
                  [sampled_masks.shape[0], 1, sampled_masks.shape[-1]]),
                jnp.concatenate([sampled_masks,
                                 jnp.tile(jnp.expand_dims(jnp.expand_dims(jnp.zeros_like(self.prior_logits), -1), 0),
                                          [sampled_masks.shape[0], 1, sampled_masks.shape[-1]])], 1))
        complement_masks = jnp.concatenate([complement_masks,
                       jnp.tile(jnp.expand_dims(jnp.expand_dims(jnp.zeros_like(self.prior_logits), -1), 0),
                                [complement_masks.shape[0], 1, complement_masks.shape[-1]])], 1)
        full_masks = jnp.concatenate([full_masks,
                       jnp.tile(jnp.expand_dims(jnp.expand_dims(jnp.zeros_like(self.prior_logits), -1), 0),
                                [full_masks.shape[0], 1, full_masks.shape[-1]])], 1)
      elif self.aggregation == 'PoE':
        #no masking of prior features
        sampled_masks = jnp.concatenate([sampled_masks,
                                 jnp.tile(jnp.expand_dims(jnp.expand_dims(jnp.ones_like(self.prior_logits), -1), 0),
                                          [sampled_masks.shape[0], 1, sampled_masks.shape[-1]])], 1)
        complement_masks = jnp.concatenate([complement_masks,
                       jnp.tile(jnp.expand_dims(jnp.expand_dims(jnp.ones_like(self.prior_logits), -1), 0),
                                [complement_masks.shape[0], 1, complement_masks.shape[-1]])], 1)
        full_masks = jnp.concatenate([full_masks,
                       jnp.tile(jnp.expand_dims(jnp.expand_dims(jnp.ones_like(self.prior_logits), -1), 0),
                                [full_masks.shape[0], 1, full_masks.shape[-1]])], 1)

    multimodal_encoding_features, multimodal_encoding_logs = self.multimodal_aggregation(
      marginal_features, full_masks)
    partial_multimodal_encoding_features, partial_multimodal_encoding_logs = self.multimodal_aggregation(
        marginal_features, sampled_masks)

    if self.aggregation in ['PoE', 'SumPooling', 'SelfAttention', 'CrossAttention']:
      multimodal_encoding_mean, multimodal_encoding_scale \
        = jnp.split(multimodal_encoding_features, 2, -1)
      partial_multimodal_encoding_mean, partial_multimodal_encoding_scale \
        = jnp.split(partial_multimodal_encoding_features, 2, -1)
      if self.posterior == 'Gaussian':
        q_dist_full = tfd.MultivariateNormalDiag(
            loc=multimodal_encoding_mean,
            scale_diag=self.min_encoder_scale + jnp.exp(multimodal_encoding_scale))
        q_dist_partial = tfd.MultivariateNormalDiag(
            loc=partial_multimodal_encoding_mean,
            scale_diag=self.min_encoder_scale + jnp.exp(partial_multimodal_encoding_scale))
      elif self.posterior == 'Laplace':
        q_dist_full = tfd.Independent(tfd.Laplace(
            loc=multimodal_encoding_mean,
            scale=self.min_encoder_scale + jnp.exp(multimodal_encoding_scale)), reinterpreted_batch_ndims=1)
        q_dist_partial = tfd.Independent(tfd.Laplace(
            loc=partial_multimodal_encoding_mean,
            scale=self.min_encoder_scale + jnp.exp(partial_multimodal_encoding_scale)), reinterpreted_batch_ndims=1)
    elif self.aggregation in ['MoE', 'SumPoolingMixture', 'SelfAttentionMixture']:
      if self.aggregation == 'MoE':
        multimodal_encoding_mean, multimodal_encoding_scale \
          = jnp.split(multimodal_encoding_features, 2, -1)
        partial_multimodal_encoding_mean, partial_multimodal_encoding_scale \
          = jnp.split(partial_multimodal_encoding_features, 2, -1)
      else:
        multimodal_encoding_mean, multimodal_encoding_scale \
          = jnp.split(multimodal_encoding_features[0].reshape([
          marginal_features.shape[0], self.K, self.T, -1]), 2, -1)
        partial_multimodal_encoding_mean, partial_multimodal_encoding_scale \
          = jnp.split(partial_multimodal_encoding_features[0].reshape([
          marginal_features.shape[0], self.K, self.T, -1]), 2, -1)
      if self.aggregation in ['SumPoolingMixture', 'SelfAttentionMixture']:
        full_mixture_weights = tfd.Categorical(logits=multimodal_encoding_features[1])
        partial_mixture_weights = tfd.Categorical(logits=partial_multimodal_encoding_features[1])
      else:
        full_mixture_weights = tfd.Categorical(
          probs=jnp.transpose(jnp.where(full_masks, 1, 0) * jnp.expand_dims(1. / jnp.sum(full_masks, 1), -1),
                              [0, 2, 1]))
        safe = jnp.where((jnp.sum(sampled_masks[:, :len(self.decoders)], 1) > 0)[:, None],
                         jnp.sum(sampled_masks[:, :len(self.decoders)], 1)[:, None],
                         1.)
        partial_probs = jnp.concatenate([
          jnp.where((jnp.sum(sampled_masks[:, :len(self.decoders)], 1) > 0)[:, None],
                    1. / safe, 0.) * sampled_masks[:, :len(self.decoders)],
          jnp.tile(jnp.exp(self.prior_logits - jax.scipy.special.logsumexp(self.prior_logits))[None, :, None],
                   [sampled_masks.shape[0], 1, sampled_masks.shape[-1]])
          * sampled_masks[:, len(self.decoders):]], 1)
        partial_mixture_weights = tfd.Categorical(probs=jax.lax.stop_gradient(jnp.transpose(partial_probs, [0, 2, 1])))

      if self.posterior == 'Gaussian':
        full_components_distribution = tfd.Independent(
          tfd.Normal(loc=jnp.transpose(multimodal_encoding_mean, [0, 2, 1, 3]),
                     scale=self.min_encoder_scale + jnp.transpose(jnp.exp(multimodal_encoding_scale), [0, 2, 1, 3])),
          reinterpreted_batch_ndims=1)
        partial_components_distribution = tfd.Independent(
          tfd.Normal(loc=jnp.transpose(partial_multimodal_encoding_mean, [0, 2, 1, 3]),
                     scale=self.min_encoder_scale + jnp.transpose(jnp.exp(partial_multimodal_encoding_scale), [0, 2, 1, 3])),
          reinterpreted_batch_ndims=1)
      elif self.posterior == 'Laplace':
        full_components_distribution = tfd.Independent(
          tfd.Laplace(loc=jnp.transpose(multimodal_encoding_mean, [0, 2, 1, 3]),
                      scale=self.min_encoder_scale + jnp.transpose(jnp.exp(multimodal_encoding_scale), [0, 2, 1, 3])),
          reinterpreted_batch_ndims=1)
        partial_components_distribution = tfd.Independent(
          tfd.Laplace(loc=jnp.transpose(partial_multimodal_encoding_mean, [0, 2, 1, 3]),
                     scale=self.min_encoder_scale + jnp.transpose(jnp.exp(partial_multimodal_encoding_scale), [0, 2, 1, 3])),
          reinterpreted_batch_ndims=1)
      q_dist_full = tfd.MixtureSameFamily(
        mixture_distribution=full_mixture_weights,
        components_distribution=full_components_distribution,
        reparameterize=True)
      q_dist_partial = tfd.MixtureSameFamily(
        mixture_distribution=partial_mixture_weights,
        components_distribution=partial_components_distribution,
        reparameterize=True)


    rng, key = random.split(rng)
    z_full = q_dist_full.sample(num_importance_samples, seed=key)
    rng, key = random.split(rng)
    z_partial = q_dist_partial.sample(num_importance_samples, seed=key)
    rng, key = random.split(rng)

    if self.stl_grad:

      if self.aggregation in ['PoE', 'SumPooling', 'SelfAttention', ]:
        if self.posterior == 'Gaussian':
          q_dist_full_stopped = tfd.MultivariateNormalDiag(
            loc=jax.lax.stop_gradient(multimodal_encoding_mean),
            scale_diag=jax.lax.stop_gradient(self.min_encoder_scale + jnp.exp(multimodal_encoding_scale)))
          q_dist_partial_stopped = tfd.MultivariateNormalDiag(
            loc=jax.lax.stop_gradient(partial_multimodal_encoding_mean),
            scale_diag=jax.lax.stop_gradient(self.min_encoder_scale + jnp.exp(partial_multimodal_encoding_scale)))
        elif self.posterior == 'Laplace':
          q_dist_full_stopped = tfd.Independent(tfd.Laplace(
            loc=jax.lax.stop_gradient(multimodal_encoding_mean),
            scale=jax.lax.stop_gradient(self.min_encoder_scale + jnp.exp(multimodal_encoding_scale))),
            reinterpreted_batch_ndims=1)
          q_dist_partial_stopped = tfd.Independent(tfd.Laplace(
            loc=jax.lax.stop_gradient(partial_multimodal_encoding_mean),
            scale=jax.lax.stop_gradient(self.min_encoder_scale + jnp.exp(partial_multimodal_encoding_scale))),
            reinterpreted_batch_ndims=1)

      elif self.aggregation in ['MoE', 'SumPoolingMixture', 'SelfAttentionMixture']:

        if self.aggregation in ['SumPoolingMixture', 'SelfAttentionMixture']:
          full_mixture_weights_stopped = tfd.Categorical(
            logits=jax.lax.stop_gradient(multimodal_encoding_features[1]))
          partial_mixture_weights_stopped = tfd.Categorical(
            logits=jax.lax.stop_gradient(partial_multimodal_encoding_features[1]))
        else:
          full_mixture_weights_stopped = full_mixture_weights
          partial_mixture_weights_stopped =partial_mixture_weights

        if self.posterior == 'Gaussian':
          full_components_distribution_stopped = tfd.Independent(
            tfd.Normal(loc=jax.lax.stop_gradient(jnp.transpose(multimodal_encoding_mean, [0, 2, 1, 3])),
                       scale=jax.lax.stop_gradient(
                         self.min_encoder_scale + jnp.transpose(jnp.exp(multimodal_encoding_scale), [0, 2, 1, 3]))),
            reinterpreted_batch_ndims=1)
          partial_components_distribution_stopped = tfd.Independent(
            tfd.Normal(loc=jax.lax.stop_gradient(jnp.transpose(partial_multimodal_encoding_mean, [0, 2, 1, 3])),
                       scale=jax.lax.stop_gradient(
                         self.min_encoder_scale + jnp.transpose(jnp.exp(partial_multimodal_encoding_scale), [0, 2, 1, 3]))),
            reinterpreted_batch_ndims=1)
        elif self.posterior == 'Laplace':
          full_components_distribution_stopped = tfd.Independent(
            tfd.Laplace(loc=jax.lax.stop_gradient(jnp.transpose(multimodal_encoding_mean, [0, 2, 1, 3])),
                        scale=jax.lax.stop_gradient(
                          self.min_encoder_scale + jnp.transpose(jnp.exp(multimodal_encoding_scale), [0, 2, 1, 3]))),
            reinterpreted_batch_ndims=1)
          partial_components_distribution_stopped = tfd.Independent(
            tfd.Laplace(loc=jax.lax.stop_gradient(jnp.transpose(partial_multimodal_encoding_mean, [0, 2, 1, 3])),
                       scale=jax.lax.stop_gradient(
                         self.min_encoder_scale + jnp.transpose(jnp.exp(partial_multimodal_encoding_scale), [0, 2, 1, 3]))),
            reinterpreted_batch_ndims=1)
        q_dist_full_stopped = tfd.MixtureSameFamily(
          mixture_distribution=full_mixture_weights_stopped,
          components_distribution=full_components_distribution_stopped,
          reparameterize=True)
        q_dist_partial_stopped = tfd.MixtureSameFamily(
          mixture_distribution=partial_mixture_weights_stopped,
          components_distribution=partial_components_distribution_stopped,
          reparameterize=True)


    c_prior_sample = tfd.Categorical(logits=self.prior_logits).sample(z_full.shape[:2], seed=key)
    prior_embedding_means_sample, prior_embedding_scales_sample = self.prior_embedding(c_prior_sample)
    if self.prior == 'Gaussian':
      dist_p_z_given_c_prior = tfd.MultivariateNormalDiag(
        loc=jnp.expand_dims(prior_embedding_means_sample, -2),
        scale_diag=jnp.expand_dims(jnp.exp(prior_embedding_scales_sample), -2))
    elif self.prior == 'Laplace':
      dist_p_z_given_c_prior = tfd.Independent(tfd.Laplace(
        loc=jnp.expand_dims(prior_embedding_means_sample, -2),
        scale=jnp.expand_dims(jnp.exp(prior_embedding_scales_sample), -2)), reinterpreted_batch_ndims=1)
    rng, key = random.split(rng)
    z_prior = dist_p_z_given_c_prior.sample(seed=key)
    rng, key = random.split(rng)


    i = 0
    rec_log_probs_partial = []
    rec_log_probs_complement = []
    rec_log_probs_full = [] #log p(x|z) under q(z|x)
    rec_log_probs_cross = [] #log p(x_cs|z)|z under q(z|x_s)
    rec_log_probs_marginal = [] #log p(x_s|z)|z under q(z|x_s)
    x_full = {}
    x_partial = {}
    x_prior = {}
    logits = {}
    for k, v in x.items():
      weight = self.rec_weights[k] if use_rec_weights else 1.
      decoder_stats_partial = jax.vmap(self.decoders[k], in_axes=0, out_axes=0)(z_partial)
      decoder_stats_full = jax.vmap(self.decoders[k], in_axes=0, out_axes=0)(z_full)
      decoder_stats_prior = jax.vmap(self.decoders[k], in_axes=0, out_axes=0)(z_prior)
      if self.x_types[k] == 'categorical':
        rec_log_prob_marginal = jnp.sum(
            jnp.where(sampled_masks[:,i,:],
                      tfd.Independent(tfd.OneHotCategorical(logits=decoder_stats_partial),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
        rec_log_probs_marginal.append(weight * rec_log_prob_marginal)
        rec_log_prob_full = jnp.sum(
            jnp.where(full_masks[:,i,:], tfd.Independent(tfd.OneHotCategorical(logits=decoder_stats_full),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
        rec_log_probs_full.append(weight * rec_log_prob_full)
        rec_log_prob_cross = jnp.sum(
            jnp.where(complement_masks[:, i, :],
                      tfd.Independent(tfd.OneHotCategorical(logits=decoder_stats_partial),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
        rec_log_probs_cross.append(weight * rec_log_prob_cross)
        if self.bound == 'masked':
          rec_log_prob_partial = rec_log_prob_marginal
          rec_log_probs_partial.append(weight * rec_log_prob_partial)
          rec_log_prob_complement = jnp.sum(
            jnp.where(complement_masks[:, i, :],
                      tfd.Independent(tfd.OneHotCategorical(logits=decoder_stats_full),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        elif self.bound == 'mixture':
          rec_log_prob_partial =rec_log_prob_marginal
          rec_log_probs_partial.append(weight * rec_log_prob_partial)
          rec_log_prob_complement = rec_log_prob_cross
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        elif self.bound == 'tc':
          rec_log_prob_partial = jnp.sum(
            jnp.where(sampled_masks[:, i, :],
                      tfd.Independent(tfd.Bernoulli(logits=decoder_stats_full),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
          rec_log_probs_partial.append(weight * rec_log_prob_partial)
          rec_log_prob_complement = jnp.sum(
            jnp.where(complement_masks[:, i, :],
                      tfd.Independent(tfd.Bernoulli(logits=decoder_stats_full),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        logits[k] = decoder_stats_full
        rng, key = random.split(rng)
        x_full[k] = tfd.OneHotCategorical(logits=decoder_stats_full).sample(seed=key)
        rng, key = random.split(rng)
        x_partial[k] = tfd.OneHotCategorical(logits=decoder_stats_partial).sample(seed=key)
        rng, key = random.split(rng)
        x_prior[k] = tfd.OneHotCategorical(logits=decoder_stats_prior).sample(seed=key)

      elif self.x_types[k] == 'bernoulli':
        rec_log_prob_marginal = jnp.sum(
            jnp.where(sampled_masks[:,i,:],
                      tfd.Independent(tfd.Bernoulli(logits=decoder_stats_partial),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
        rec_log_probs_marginal.append(weight * rec_log_prob_marginal)
        rec_log_prob_full = jnp.sum(
            jnp.where(full_masks[:,i,:],
                      tfd.Independent(tfd.Bernoulli(logits=decoder_stats_full),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
        rec_log_probs_full.append(weight * rec_log_prob_full)
        rec_log_prob_cross = jnp.sum(
            jnp.where(complement_masks[:, i, :],
                      tfd.Independent(tfd.Bernoulli(logits=decoder_stats_partial),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
        rec_log_probs_cross.append(weight * rec_log_prob_cross)
        if self.bound == 'masked':
          rec_log_probs_partial.append(weight * rec_log_prob_marginal)
          rec_log_prob_complement = jnp.sum(
            jnp.where(complement_masks[:, i, :],
                      tfd.Independent(tfd.Bernoulli(logits=decoder_stats_full),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        elif self.bound == 'mixture':
          rec_log_prob_partial = rec_log_prob_marginal
          rec_log_probs_partial.append(weight * rec_log_prob_partial)
          rec_log_prob_complement =rec_log_prob_cross
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        elif self.bound == 'tc':
          rec_log_prob_partial = jnp.sum(
            jnp.where(sampled_masks[:, i, :],
                      tfd.Independent(tfd.Bernoulli(logits=decoder_stats_full),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
          rec_log_probs_partial.append(weight * rec_log_prob_partial)
          rec_log_prob_complement = jnp.sum(
            jnp.where(complement_masks[:, i, :],
                      tfd.Independent(tfd.Bernoulli(logits=decoder_stats_full),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        rng, key = random.split(rng)
        x_full[k] = tfd.Bernoulli(logits=decoder_stats_full).sample(seed=key)
        rng, key = random.split(rng)
        x_partial[k] = tfd.Bernoulli(logits=decoder_stats_partial).sample(seed=key)
        rng, key = random.split(rng)
        x_prior[k] = tfd.Bernoulli(logits=decoder_stats_prior).sample(seed=key)

      elif self.x_types[k] == 'normal':
        loc_partial, scale_partial = decoder_stats_partial
        loc_full, scale_full = decoder_stats_full
        loc_prior, scale_prior = decoder_stats_prior
        rec_log_prob_marginal = jnp.sum(
            jnp.where(sampled_masks[:,i,:],
                      tfd.MultivariateNormalDiag(
                        loc=loc_partial, scale_diag=self.min_decoder_scale + jnp.exp(scale_partial)).log_prob(x[k]),0), -1)
        rec_log_probs_marginal.append(weight * rec_log_prob_marginal)
        rec_log_prob_full = jnp.sum(
            jnp.where(full_masks[:,i,:],
                      tfd.MultivariateNormalDiag(
          loc=loc_full, scale_diag=self.min_decoder_scale + jnp.exp(scale_full)).log_prob(x[k]), 0), -1)
        rec_log_probs_full.append(weight * rec_log_prob_full)
        rec_log_prob_cross = jnp.sum(
            jnp.where(complement_masks[:,i,:],
                      tfd.MultivariateNormalDiag(
                        loc=loc_partial, scale_diag=self.min_decoder_scale + jnp.exp(scale_partial)).log_prob(x[k]),0), -1)
        rec_log_probs_cross.append(weight * rec_log_prob_cross)
        if self.bound == 'masked':
          rec_log_probs_partial.append(weight * rec_log_prob_marginal)
          rec_log_prob_complement = jnp.sum(
            jnp.where(complement_masks[:,i,:],
                      tfd.MultivariateNormalDiag(
                        loc=loc_full, scale_diag=self.min_decoder_scale + jnp.exp(scale_full)).log_prob(x[k]),0), -1)
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        elif self.bound == 'mixture':
          rec_log_prob_partial = rec_log_prob_marginal
          rec_log_probs_partial.append(weight * rec_log_prob_partial)
          rec_log_prob_complement = rec_log_prob_cross
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        elif self.bound == 'tc':
          rec_log_prob_partial = jnp.sum(
            jnp.where(sampled_masks[:, i, :],
                      tfd.MultivariateNormalDiag(
                        loc=loc_full, scale_diag=self.min_decoder_scale + jnp.exp(scale_full)).log_prob(x[k]), 0), -1)
          rec_log_probs_partial.append(weight * rec_log_prob_partial)
          rec_log_prob_complement = jnp.sum(
            jnp.where(complement_masks[:, i, :],
                      tfd.MultivariateNormalDiag(
                        loc=loc_full, scale_diag=self.min_decoder_scale + jnp.exp(scale_full)).log_prob(x[k]), 0), -1)
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        x_full[k] = tfd.MultivariateNormalDiag(loc=loc_full, scale_diag=self.min_decoder_scale + jnp.exp(scale_full)).sample(seed=key)
        rng, key = random.split(rng)
        x_partial[k] = tfd.MultivariateNormalDiag(loc=loc_partial, scale_diag=self.min_decoder_scale + jnp.exp(scale_partial)).sample(seed=key)
        rng, key = random.split(rng)
        x_prior[k] = tfd.MultivariateNormalDiag(loc=loc_prior, scale_diag=jnp.exp(scale_prior)).sample(seed=key)

      elif self.x_types[k] == 'laplace':
        #d_k = x[k].shape[-1]
        loc_partial, scale_partial = decoder_stats_partial
        loc_full, scale_full = decoder_stats_full
        loc_prior, scale_prior = decoder_stats_prior
        rec_log_prob_marginal = jnp.sum(
            jnp.where(sampled_masks[:,i,:],
                      tfd.Independent(tfd.Laplace(loc=loc_partial, scale=self.min_decoder_scale + jnp.exp(scale_partial)),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
        rec_log_probs_marginal.append(weight * rec_log_prob_marginal)
        rec_log_prob_full = jnp.sum(
            jnp.where(full_masks[:,i,:],
                      tfd.Independent(tfd.Laplace(loc=loc_full, scale=self.min_decoder_scale + jnp.exp(scale_full)),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
        rec_log_probs_full.append(weight * rec_log_prob_full)
        rec_log_prob_cross = jnp.sum(
            jnp.where(complement_masks[:,i,:],
                      tfd.Independent(tfd.Laplace(loc=loc_partial, scale=self.min_decoder_scale + jnp.exp(scale_partial)),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
        rec_log_probs_cross.append(weight * rec_log_prob_cross)
        if self.bound == 'masked':
          rec_log_probs_partial.append(weight * rec_log_prob_marginal)
          rec_log_prob_complement = jnp.sum(
            jnp.where(complement_masks[:, i, :],
                      tfd.Independent(tfd.Laplace(loc=loc_full, scale=self.min_decoder_scale + jnp.exp(scale_full)),
                                      reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        elif self.bound == 'mixture':
          rec_log_prob_partial = rec_log_prob_marginal
          rec_log_probs_partial.append(weight * rec_log_prob_partial)
          rec_log_prob_complement = rec_log_prob_cross
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        elif self.bound == 'tc':
          rec_log_prob_partial = jnp.sum(
            jnp.where(sampled_masks[:, i, :],
                      tfd.Independent(tfd.Laplace(
                        loc=loc_full, scale=self.min_decoder_scale + jnp.exp(scale_full)),
                        reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
          rec_log_probs_partial.append(weight * rec_log_prob_partial)
          rec_log_prob_complement = jnp.sum(
            jnp.where(complement_masks[:, i, :],
                      tfd.Independent(tfd.Laplace(
                        loc=loc_full, scale=self.min_decoder_scale + jnp.exp(scale_full)),
                        reinterpreted_batch_ndims=1).log_prob(x[k]), 0), -1)
          rec_log_probs_complement.append(weight * rec_log_prob_complement)

        rng, key = random.split(rng)
        x_full[k] = tfd.Laplace(loc=loc_full, scale=self.min_decoder_scale + jnp.exp(scale_full)).sample(seed=key)
        rng, key = random.split(rng)
        x_partial[k] = tfd.Laplace(loc=loc_partial, scale=self.min_decoder_scale + jnp.exp(scale_partial)).sample(seed=key)
        rng, key = random.split(rng)
        x_prior[k] = tfd.Laplace(loc=loc_prior, scale=jnp.exp(scale_prior)).sample(seed=key)


      i = i + 1

    #sum reconstruction terms over modalities
    rec_log_probs_partial = jnp.sum(jnp.stack(rec_log_probs_partial), 0)
    rec_log_probs_complement = jnp.sum(jnp.stack(rec_log_probs_complement), 0)
    rec_log_probs_full = jnp.sum(jnp.stack(rec_log_probs_full), 0)
    rec_log_probs_marginal = jnp.sum(jnp.stack(rec_log_probs_marginal), 0)
    rec_log_probs_cross = jnp.sum(jnp.stack(rec_log_probs_cross), 0)

    if not self.stl_grad:
      log_q_z_full = jnp.sum(q_dist_full.log_prob(z_full), -1)
      log_q_z_partial = jnp.sum(q_dist_partial.log_prob(z_partial), -1)
    else:
      log_q_z_full = jnp.sum(q_dist_full_stopped.log_prob(z_full), -1)
      log_q_z_partial = jnp.sum(q_dist_partial_stopped.log_prob(z_partial), -1)
    log_q_z_partial_at_full = jnp.sum(q_dist_partial.log_prob(z_full), -1)


    log_p_c = self.prior_logits - jax.scipy.special.logsumexp(self.prior_logits)
    cs = jnp.arange(self.K)
    prior_embedding_means, prior_embedding_scales = self.prior_embedding(jnp.expand_dims(cs, 0))
    if self.prior == 'Gaussian':
      dist_p_z_given_cs = tfd.Independent(tfd.Normal(
        loc=jnp.squeeze(prior_embedding_means, 0),
        scale=jnp.exp(jnp.squeeze(prior_embedding_scales))), reinterpreted_batch_ndims=1)
    elif self.prior == 'Laplace':
      dist_p_z_given_cs =tfd.Independent(tfd.Laplace(
        loc=jnp.squeeze(prior_embedding_means, 0),
        scale=jnp.exp(jnp.squeeze(prior_embedding_scales))), reinterpreted_batch_ndims=1)

    log_p_z_partial_given_c = dist_p_z_given_cs.log_prob(jnp.expand_dims(z_partial, -2))
    log_p_z_partial = jax.scipy.special.logsumexp(jnp.sum(log_p_z_partial_given_c, -2) + log_p_c, -1)
    log_p_z_full_given_c = dist_p_z_given_cs.log_prob(jnp.expand_dims(z_full, -2))
    log_p_z_full = jax.scipy.special.logsumexp(jnp.sum(log_p_z_full_given_c, -2) + log_p_c, -1)

    if self.bound == 'masked':
      f_x = rec_log_probs_partial - self.beta * log_q_z_partial
      f_c = self.beta * log_p_z_partial
      g_x = rec_log_probs_complement - self.beta * log_q_z_full + self.beta * log_q_z_partial_at_full
    elif self.bound == 'mixture':
      f_x = rec_log_probs_partial + rec_log_probs_complement - self.beta * log_q_z_partial
      f_c = self.beta * log_p_z_partial
      g_x = jnp.zeros_like(f_x)
    elif self.bound == 'tc':
      f_x = rec_log_probs_partial + rec_log_probs_complement - self.beta * log_q_z_full + \
            self.beta * log_q_z_partial_at_full
      f_c = jnp.zeros_like(f_x)
      g_x = jnp.zeros_like(f_x)

    #entropy approximation of prior term
    log_p_z_prior_given_c = dist_p_z_given_cs.log_prob(jnp.expand_dims(z_prior, -2))
    if self.T == 1:
      log_p_z_prior = jax.scipy.special.logsumexp(log_p_z_prior_given_c  + log_p_c, -1)
    else:
      log_p_z_prior = jax.scipy.special.logsumexp(jnp.sum(log_p_z_prior_given_c, 1) + log_p_c, -1)

    log_p_c_given_z_full = log_p_c[None,None,None,] + log_p_z_full_given_c
    log_p_c_given_z_full = log_p_c_given_z_full - jnp.expand_dims(
      jax.scipy.special.logsumexp(log_p_c_given_z_full, -1), -1)


    return f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement,\
           log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, logits, log_p_z_prior,\
           rec_log_probs_full, rec_log_probs_marginal, rec_log_probs_cross, x_full, x_partial, x_prior, log_p_z_full, \
           multimodal_encoding_features, z_full, log_p_c_given_z_full, \
           multimodal_encoding_logs, partial_multimodal_encoding_logs, sampled_masks, rng

  def train(self, state, x, x_mask, rng, steps, num_importance_samples=1,
            batch_size=None, max_iter=None, log_dir=None, checkpoint_dir=None):

    if log_dir is not None:
      if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
      log_dir_writer = os.path.join(log_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
      if not os.path.exists(log_dir_writer): os.makedirs(log_dir_writer, exist_ok=True)
      writer = tensorboard.SummaryWriter(log_dir_writer)
    if checkpoint_dir is not None:
      if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir, exist_ok=True)


    def loss_fn(params, x, x_mask, rng):
      f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement, \
      log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, logits, log_p_z_prior, \
      rec_log_probs_full, rec_log_probs_marginal, rec_log_probs_cross, x_full, x_partial, x_prior, \
      log_p_z_full, multi_modal_encoding_features,\
      _, _, multimodal_encoding_logs, partial_multimodal_encoding_logs, sampled_masks, rng = self.apply(
        {'params': params}, x, x_mask, num_importance_samples, rng, use_rec_weights=True)
      loss = - jax.scipy.special.logsumexp(f_x + f_c + g_x, 0)
      return jnp.sum(loss), (f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement,
                             rec_log_probs_full, rec_log_probs_marginal, rec_log_probs_cross,
                             log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, logits,
                             log_p_z_prior, x_full, x_partial, x_prior, log_p_z_full,
                             multimodal_encoding_logs, partial_multimodal_encoding_logs, sampled_masks, rng)
    @jax.jit
    def train_step(state, x_batch, x_mask_batch, rng):
      params = state.params
      loss_value, grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, x_batch, x_mask_batch, rng)
      return state, grads, loss_value

    for epoch in range(steps):
      rng, key = random.split(rng)

      if batch_size is None:
        state, grad, loss = train_step(state, x, x_mask, key)
        state = state.apply_gradients(grads=grad)
        f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement, \
        rec_log_probs_full, rec_log_probs_marginal, rec_log_probs_cross, \
        log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, logits, log_p_z_prior,\
        x_full, x_partial, x_prior, log_p_z_full,  multimodal_encoding_logs, partial_multimodal_encoding_logs,\
        sampled_masks, rng \
          = loss[1]
        marginal_bound = jnp.sum(rec_log_probs_partial + log_p_z_partial - log_q_z_partial)
        conditional_bound = jnp.sum(rec_log_probs_complement + log_q_z_partial_at_full - log_q_z_full)
        neg_kl_full_partial = jnp.sum(log_q_z_partial_at_full - log_q_z_full)
        neg_kl_partial_prior = jnp.sum(log_p_z_partial - log_q_z_partial)
        reconstruction_full = jnp.sum(rec_log_probs_full)
        reconstruction_marginal = jnp.sum(rec_log_probs_marginal)
        reconstruction_cross = jnp.sum*rec_log_probs_cross
        epoch_loss = loss[0]
        accuracy = {}
        for k in x.keys():
          if self.x_types[k]=='categorical':
            true_pos = jnp.sum(jnp.where(x_mask[k], jnp.argmax(logits[k], -1) == jnp.argmax(x[k], -1), 0))
            accuracy[k] = true_pos/jnp.sum(x_mask[k])

      else:
        epoch_loss = []
        accuracies = {k: [] for k in self.x_types.keys() if self.x_types[k]=='categorical'}
        marginal_bounds = []
        conditional_bounds = []
        neg_kls_full_partial = []
        neg_kls_partial_prior = []
        reconstructions_full = []
        reconstructions_marginal = []
        reconstructions_cross = []
        log_p_z_priors = []
        if not self.use_data_loader:
          ds_size = list(x.values())[0].shape[:-1]
          steps_per_epoch = ds_size[0] // batch_size
          perms = random.permutation(rng, ds_size[0])
          perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
          perms = perms.reshape((steps_per_epoch, batch_size))
        else:
          perms = iter(x)

        step = 0
        print(len(perms))
        for perm in perms:
          if not self.use_data_loader:
            batch_x = {k: v[perm, ...] for k,v in x.items()}
            batch_x_mask = {k: v[perm, ...] for k,v in x_mask.items()}
          else:
            batch_x, batch_x_mask = self.collate_fn(perm)
          state, grad, loss = train_step(state, batch_x, batch_x_mask, key)
          state = state.apply_gradients(grads=grad)
          print((step,loss[0]))
          f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement, \
          rec_log_probs_full, rec_log_probs_marginal, rec_log_probs_cross, \
          log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, \
          logits, log_p_z_prior, x_full, x_partial, x_prior, log_p_z_full, \
          multimodal_encoding_logs, partial_multimodal_encoding_logs, sampled_masks, rng = loss[1]
          epoch_loss.append(loss[0])
          marginal_bound = jnp.sum(rec_log_probs_partial + log_p_z_partial - log_q_z_partial)
          conditional_bound = jnp.sum(rec_log_probs_complement + log_q_z_partial_at_full - log_q_z_full)
          neg_kl_full_partial = jnp.sum(log_q_z_partial_at_full - log_q_z_full)
          neg_kl_partial_prior = jnp.sum(log_p_z_partial - log_q_z_partial)
          reconstruction_full = jnp.sum(rec_log_probs_full)
          reconstruction_marginal = jnp.sum(rec_log_probs_marginal)
          reconstruction_cross = jnp.sum(rec_log_probs_cross)
          marginal_bounds.append(marginal_bound)
          conditional_bounds.append(conditional_bound)
          neg_kls_full_partial.append(neg_kl_full_partial)
          neg_kls_partial_prior.append(neg_kl_partial_prior)
          reconstructions_full.append(reconstruction_full)
          reconstructions_marginal.append(reconstruction_marginal)
          reconstructions_cross.append(reconstruction_cross)
          log_p_z_priors.append(jnp.sum(log_p_z_prior))
          for k in self.x_types.keys():
            if self.x_types[k] == 'categorical':
              #sample index of used importance sample
              rng, key = random.split(rng)
              j = tfd.Categorical(logits=jnp.transpose(f_x + f_c + g_x)).sample(seed=key)
              selected_logits = jnp.squeeze(jnp.take_along_axis(logits[k],j[None,:,None,None,None], axis=0), 0)
              true_pos =jnp.sum(jnp.where(jnp.expand_dims(batch_x_mask[k], -1),
                                          jnp.argmax(selected_logits, -1) == jnp.argmax(batch_x[k], -1), 0))
              accuracies[k].append(true_pos / jnp.sum(batch_x_mask[k]))

          step = step+1
          if max_iter is not None:
            if step > max_iter:
              break


        if log_dir is not None:
          for k, v in state.params.items():
            parameters = jax.tree_util.tree_leaves(v)#jax.tree_leaves(v)
            for i in range(len(parameters)):
              writer.histogram('params'+k+'/'+str(i), jax.device_get(parameters[i]), state.step)  # jax.device_get(query_seed[0]), t[0][0])

          for k, v in multimodal_encoding_logs.items():
            if k[1] == 'scalar':
              writer.scalar('full_aggregation/'+k[0], jax.device_get(v), state.step)  # jax.device_get(query_seed[0]), t[0][0])
            elif k[1] == 'vector':
              writer.histogram('full_aggregation/'+k[0], jax.device_get(v), state.step)  # jax.device_get(query_seed[0]), t[0][0])
          for k, v in partial_multimodal_encoding_logs.items():
            if k[1] == 'scalar':
              writer.scalar('partial_aggregation/'+k[0], jax.device_get(v), state.step)  # jax.device_get(query_seed[0]), t[0][0])
            elif k[1] == 'vector':
              writer.histogram('partial_aggregation/'+k[0], jax.device_get(v), state.step)  # jax.device_get(query_seed[0]), t[0][0])

        epoch_loss = jnp.sum(jnp.stack(epoch_loss))
        marginal_bound = jnp.sum(jnp.stack(marginal_bounds))
        conditional_bound = jnp.sum(jnp.stack(conditional_bounds))
        neg_kl_partial_prior = jnp.sum(jnp.stack(neg_kls_partial_prior))
        neg_kl_full_partial = jnp.sum(jnp.stack(neg_kls_full_partial))
        reconstruction_full = jnp.sum(jnp.stack(reconstructions_full))
        reconstruction_marginal = jnp.sum(jnp.stack(reconstructions_marginal))
        reconstruction_cross = jnp.sum(jnp.stack(reconstructions_cross))
        accuracy = {k: jnp.mean(jnp.stack(v)) for k,v in accuracies.items()}

        if log_dir is not None:
          writer.scalar('loss/loss', jax.device_get(epoch_loss), state.step)
          writer.scalar('loss/marginal_bound', jax.device_get(marginal_bound), state.step)
          writer.scalar('loss/conditional_bound', jax.device_get(conditional_bound), state.step)
          writer.scalar('loss/neg_kl_partial_prior', jax.device_get(neg_kl_partial_prior), state.step)
          writer.scalar('loss/neg_kl_full_partial', jax.device_get(neg_kl_full_partial), state.step)
          writer.scalar('loss/reconstruction_full', jax.device_get(reconstruction_full), state.step)
          writer.scalar('loss/reconstruction_marginal', jax.device_get(reconstruction_marginal), state.step)
          writer.scalar('loss/reconstruction_cross', jax.device_get(reconstruction_cross), state.step)

      print('step:% 3d, train_loss: %.4f,  train_marginal_bound: %.4f,  train_conditional_bound: %.4f,'
            '  train_neg_kl_full_partial: %.4f, train_neg_kl_partial_prior: %.4f, train_reconstruction_full: %.4f,'
            'train_reconstruction_marginal: %.4f, ,train_reconstruction_cross: %.4f'% (
              state.step, epoch_loss, marginal_bound, conditional_bound, neg_kl_full_partial, neg_kl_partial_prior,
              reconstruction_full, reconstruction_marginal, reconstruction_cross))
      for k,v in self.x_types.items():
        if v == 'categorical':
          print('modality {} reconstruction accuracy: {}', (k, accuracy[k]))

      if checkpoint_dir is not None:
        checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=state, step=state.step,
                                  overwrite=True, keep=5, keep_every_n_steps=10 * len(perms))
        np.savetxt(os.path.join(checkpoint_dir, 'loss__{}.txt'.format(state.step)), jnp.array([epoch_loss]))

    return state, grad, epoch_loss, rng


  def llh_eval(self, state, x, x_mask, batch_size, rng, num_importance_samples=100,
               max_iter=None, collate_fn=None):
    if collate_fn is None: collate_fn = self.collate_fn
    #@partial(jax.jit, static_argnames=['num_impportance_samples'])
    @jax.jit
    def llh_step(state, x_batch, x_mask_batch, rng):
      params = state.params
      f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement, \
        log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, logits, log_p_z_prior,\
        recon_log_prob_full, recon_log_prob_marginal, recon_log_prob_cross,\
        x_full, x_partial, x_prior, log_p_z_full, _, _, _, _, _,_, rng = self.apply(
          {'params': params}, x_batch, x_mask_batch, num_importance_samples=num_importance_samples, rng=rng,
        mask_prob=0.)
      joint_llh = jax.scipy.special.logsumexp(recon_log_prob_full + log_p_z_full - log_q_z_full, 0) - jnp.log(
        num_importance_samples)
      return joint_llh, rng

    if not self.use_data_loader:
      ds_size = list(x.values())[0].shape[:-1]
      steps_per_epoch = ds_size[0] // batch_size
      perms = random.permutation(rng, ds_size[0])
      perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
      perms = perms.reshape((steps_per_epoch, batch_size))
    else:
      perms = iter(x)
    joint_llh = []
    i = 0
    for perm in perms:
      if not self.use_data_loader:
        batch_x = {k: v[perm, ...] for k, v in x.items()}
        batch_x_mask = {k: v[perm, ...] for k, v in x_mask.items()}
      else:
        batch_x, batch_x_mask = collate_fn(perm)
      joint_llh_, rng = llh_step(state, batch_x, batch_x_mask, rng)
      joint_llh.append(joint_llh_)
      del joint_llh_
      i = i + 1
      if max_iter is not None:
        if i > max_iter:
          break
    return jnp.concatenate(joint_llh), rng


  def classifier_eval(self, state, x, batch_size, rng, max_iter=None,
                      collate_fn_cond=None, collate_fn_eval=None, x_mask_cond=None, x_mask_eval=None):

    @jax.jit
    def classifier_step(state, x_batch, x_mask_cond_batch, x_mask_eval_batch, rng):
      params = state.params
      f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement, \
      log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, logits, log_p_z_prior, \
      recon_log_prob_full, _, _, x_full, x_partial, x_prior, log_p_z_full, multi_modal_encoding_features, _, _, _, _, _, rng = \
        self.apply(
        {'params': params}, x_batch, x_mask_cond_batch, mask_prob=0., rng=rng, num_importance_samples=1)
      true_pos = {}
      number_labels = {}
      for k in self.x_types.keys():
        if self.x_types[k] == 'categorical':
          true_pos[k] = jnp.sum(jnp.where(x_mask_eval_batch[k], jnp.argmax(logits[k], -1) == jnp.argmax(x_batch[k], -1), 0))
          number_labels[k] = jnp.sum(x_mask_eval_batch[k])
      return true_pos, number_labels, rng


    true_pos = []
    number_labels = []
    if not self.use_data_loader:
      ds_size = list(x.values())[0].shape[:-1]
      steps_per_epoch = ds_size[0] // batch_size
      perms = random.permutation(rng, ds_size[0])
      perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
      perms = perms.reshape((steps_per_epoch, batch_size))
    else:
      perms = iter(x)
    if collate_fn_cond is None:
      x_mask_cond = iter(x_mask_cond)
    i = 0
    for perm in perms:
      if not self.use_data_loader:
        batch_x = {k: v[perm, ...] for k, v in x.items()}
        batch_x_mask_cond = {k: v[perm, ...] for k, v in x_mask_cond.items()}
        batch_x_mask_eval = {k: v[perm, ...] for k, v in x_mask_eval.items()}
      else:
        if collate_fn_eval is None:
          batch_x, batch_x_mask_eval = self.collate_fn_eval(perm)
        else:
          batch_x, batch_x_mask_eval = collate_fn_eval(perm)
        if collate_fn_cond is None:
          _, batch_x_mask_cond = self.collate_fn_cond(next(x_mask_cond))
        else:
          _, batch_x_mask_cond = collate_fn_cond(perm)
      true_pos_, number_labels_, rng = classifier_step(state, batch_x, batch_x_mask_cond, batch_x_mask_eval, rng)
      true_pos.append(true_pos_)
      number_labels.append(number_labels_)
      i = i + 1
      if max_iter is not None:
        if i > max_iter:
          break
    return (
      {k: np.sum([sub[k] for sub in true_pos]) for k in true_pos[0].keys()},
      {k: np.sum([sub[k] for sub in number_labels]) for k in number_labels[0].keys()},
      rng)
  def encode(self, state, x, x_mask,rng, mask_prob=0.):
    params = state.params
    f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement, \
    log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, logits, log_p_z_prior, \
    recon_log_prob_full, recon_log_prob_marginal, recon_log_prob_cross, x_full, x_partial, x_prior, log_p_z_full,\
    multi_modal_encoding_features, _, _, _, _, sampled_masks, rng =\
      self.apply({'params': params}, x, x_mask,  mask_prob=mask_prob, rng=rng, num_importance_samples=1)
    return multi_modal_encoding_features, recon_log_prob_full, recon_log_prob_marginal, recon_log_prob_cross,\
      log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, log_p_z_full, sampled_masks

  def encode_dataloader(self, state, x, rng, collate_fn=None):

    @jax.jit
    def encode_step(params, x, x_mask, rng, mask_prob=0.):
      f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement, \
      log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, logits, log_p_z_prior, \
      recon_log_prob_full, recon_log_prob_marginal, recon_log_prob_cross, x_full, x_partial, x_prior, log_p_z_full,\
      multi_modal_encoding_features, _, _, _, _, sampled_masks, rng =\
        self.apply({'params': params}, x, x_mask,  mask_prob=mask_prob, rng=rng, num_importance_samples=1)
      return multi_modal_encoding_features, recon_log_prob_full, recon_log_prob_marginal, recon_log_prob_cross,\
        log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, log_p_z_full, sampled_masks

    perms = iter(x)
    recon_log_probs_full = []
    recon_log_probs_cross = []
    log_q_z_fulls = []
    log_q_z_partial_at_fulls = []
    log_p_z_fulls = []

    if collate_fn is None:
      collate_fn = self.collate_fn
    for perm in perms:
      batch_x, batch_x_mask_cond = collate_fn(perm)
      multi_modal_encoding_features, recon_log_prob_full, recon_log_prob_marginal, recon_log_prob_cross, \
      log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, log_p_z_full, sampled_masks \
      = encode_step(state.params, batch_x, batch_x_mask_cond, rng, mask_prob=None)
      rng, key = random.split(rng)
      recon_log_probs_full.append(recon_log_prob_full)
      recon_log_probs_cross.append(recon_log_prob_cross)
      log_q_z_fulls.append(log_q_z_full)
      log_q_z_partial_at_fulls.append(log_q_z_partial_at_full)
      log_p_z_fulls.append(log_p_z_full)

    return (
      jnp.concatenate([sub for sub in recon_log_probs_full], 1),
      jnp.concatenate([sub for sub in recon_log_probs_cross], 1),
      jnp.concatenate([sub for sub in log_q_z_fulls], 1),
      jnp.concatenate([sub for sub in log_q_z_partial_at_fulls], 1),
      jnp.concatenate([sub for sub in log_p_z_fulls], 1),
      rng)

  def generate(self, state, x, x_mask_cond, batch_size, rng, num_importance_samples=1, max_iter=None,
               collate_fn=None, data_loader=None, save_img=False):

    @jax.jit
    def generate_step(state, x_batch, x_mask_cond_batch, rng):
      params = state.params
      f_x, f_c, g_x, rec_log_probs_partial, rec_log_probs_complement, \
      log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, logits, log_p_z_prior, \
      recon_log_prob_full, _, _, x_full, x_partial, x_prior, log_p_z_full, multi_modal_encoding_features, \
      z_full, log_p_c_given_z_full, _, _, _, rng = self.apply(
        {'params': params}, x_batch, x_mask_cond_batch,
        num_importance_samples=num_importance_samples, rng=rng, mask_prob=0.)
      return x_full, x_prior, z_full, log_p_c_given_z_full, rng


    if data_loader is not None:
      #os.makedirs(os.path.join(data_loader, 'generated', 'test'), exist_ok=True)
      os.makedirs(os.path.join(data_loader, 'generated','x'), exist_ok=True)
      os.makedirs(os.path.join(data_loader, 'generated','x_prior'), exist_ok=True)
      os.makedirs(os.path.join(data_loader, 'generated','z'), exist_ok=True)
      os.makedirs(os.path.join(data_loader, 'generated','log_p_c'), exist_ok=True)
      os.makedirs(os.path.join(data_loader, 'generated','true_x'), exist_ok=True)


    x_full = []
    x_prior = []
    z_full = []
    i = 0
    log_p_c_given_z_full = []
    if not self.use_data_loader:
      ds_size = list(x.values())[0].shape[:-1]
      steps_per_epoch = ds_size[0] // batch_size
      perms = random.permutation(rng, ds_size[0])
      perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
      perms = perms.reshape((steps_per_epoch, batch_size))
    else:
      perms = iter(x)
    for perm in perms:
      if not self.use_data_loader:
        batch_x = {k: v[perm, ...] for k, v in x.items()}
        batch_x_mask_cond = {k: v[perm, ...] for k, v in x_mask_cond.items()}
      else:
        if collate_fn is None:
          collate_fn = self.collate_fn
        batch_x, batch_x_mask_cond = collate_fn(perm)
      x_full_, x_prior_, z_full_, log_p_c_, rng = generate_step(state, batch_x, batch_x_mask_cond, rng)
      if not data_loader:
        x_full.append(x_full_)
        x_prior.append(x_prior_)
        z_full.append(z_full_)
        log_p_c_given_z_full.append(log_p_c_)
      else:
        np.save(os.path.join(data_loader, 'generated', 'x', str(i)+'.npy'), x_full_)
        np.save(os.path.join(data_loader, 'generated', 'x_prior', str(i)+'.npy'), x_prior_)
        np.save(os.path.join(data_loader, 'generated', 'z', str(i)+'.npy'), z_full_)
        np.save(os.path.join(data_loader, 'generated', 'log_p_c', str(i)+'.npy'), log_p_c_)
        np.save(os.path.join(data_loader, 'generated', 'true_x', str(i)+'.npy'), batch_x)

      i = i + 1
      if max_iter is not None:
        if i > max_iter:
          break
    if not data_loader:
      return (
        {k: jnp.concatenate([sub[k] for sub in x_full], 1) for k in x_full[0].keys()},
        {k: jnp.concatenate([sub[k] for sub in x_prior], 1) for k in x_prior[0].keys()},
        jnp.concatenate([sub for sub in z_full], 1),
        jnp.concatenate([sub for sub in log_p_c_given_z_full], 1),
        perms,
        rng)
