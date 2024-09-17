import numpy as np
from absl import app
import jax.numpy as jnp
from absl import flags
import sys
import pandas as pd
import jax.numpy as jnp
from jax import random
from multimodal_vae import MultiModalVAE
import optax
import os
from flax import linen as nn
import jax
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from flax.training import train_state, checkpoints


FLAGS = flags.FLAGS
flags.DEFINE_string(
  'bound', default='masked',
  help=('bound type (masked, mixture, tc')
)
flags.DEFINE_integer(
    'seed', default=1,
    help=('seed for rng')
)
flags.DEFINE_integer(
    'latent_dim', default=5,
    help=('dimension of latent state')
)
flags.DEFINE_integer(
    'x_dim_max', default=60,
    help=('max dimension of observation')
)
flags.DEFINE_integer(
    'num_modalities', default=5,
    help=('number of modalities')
)
flags.DEFINE_integer(
    'train_size', default=5000,
    help=('train size for each modality')
)
flags.DEFINE_integer(
    'batch_size', default=500,
    help=('batch size')
)
flags.DEFINE_integer(
    'training_epochs', default=4000,
    help=('training epochs')
)
flags.DEFINE_float(
    'beta', default=1.,
    help=('beta for weighting rate term')
)
flags.DEFINE_float(
    'learning_rate', default=.001,
    help=('learning rate')
)
flags.DEFINE_bool(
    'sparse', default=True,
    help=('whether to use a sparse loading matrix for modality-specific components')
)
flags.DEFINE_string(
    'aggregation', default='SumPooling',
    help=('multi modal aggregation scheme (PoE, SumPooling, Attention.')
)
flags.DEFINE_string(
  'output_dir', default=os.getcwd(),
  help=('output directory for results,')
)
flags.DEFINE_bool(
  'estimate_model', default=True,
  help=('whether to estimate the decoder model,')
)
flags.DEFINE_integer(
    'pooling_size', default=256,
    help=('pooling size')
)
flags.DEFINE_integer(
  'aggregation_size', default=256,
  help=('MLP hidden layer dimensions for aggregation or attention size')
)
flags.DEFINE_integer(
  'encoder_dim', default=256,
  help=('dimension of hidden layers for modality-specific encoders')
)
flags.DEFINE_integer(
  'pooling_aggregation_depth', default=2,
  help=('hidden layer depth for pooling nn')
)
def main(argv):
  del argv  # Unused.


  FLAGS(sys.argv)
  output_dir = os.path.join(
    FLAGS.output_dir, str(FLAGS.sparse), str(FLAGS.num_modalities), str(FLAGS.beta), str(FLAGS.latent_dim),
    str(FLAGS.bound), str(FLAGS.aggregation), str(FLAGS.seed))

  if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  flag_file = open(os.path.join(output_dir, 'flags.txt'), "w")
  flag_file.write(s)
  flag_file.close()


  rng = random.PRNGKey(FLAGS.seed)
  rng, key = random.split(rng)
  #simulate data generating parameters
  modalities = [str(i) for i in range(FLAGS.num_modalities)]
  if not FLAGS.sparse:
    latent_dim = FLAGS.latent_dim
  else:
    latent_dim = FLAGS.latent_dim * (1 + FLAGS.num_modalities)
  T = 1
  x_types = {k: 'normal' for k in modalities}


  train_size = FLAGS.train_size
  x_dims_vector = jax.random.randint(key=key, minval=FLAGS.x_dim_max//2, maxval=FLAGS.x_dim_max, shape=[FLAGS.num_modalities])
  x_dims = {str(k):x_dims_vector[k] for k in range(FLAGS.num_modalities)}
  rng, key = random.split(rng)
  sigma_squared_x = 1.
  rng, key = random.split(rng)
  #sample decoder model
  W = {}
  b = {}
  for s in modalities:
    if not FLAGS.sparse:
      W[s] = np.linalg.qr(tfd.Uniform(
        low=-1*np.ones([x_dims[s], latent_dim]), high=np.ones([x_dims[s], latent_dim])).sample(seed=key))[0]
      rng, key = random.split(rng)
      b[s] = tfd.Normal(loc=jnp.zeros([x_dims[s]]), scale=1.).sample(seed=key)
      rng, key = random.split(rng)
    else:
      #jointly orthogonal
      W_ = np.linalg.qr(tfd.Uniform(
        low=-1*np.ones([x_dims[s], 2*FLAGS.latent_dim]), high=np.ones([x_dims[s], 2*FLAGS.latent_dim])).sample(seed=key))[0]
      W_p = W_[:,:FLAGS.latent_dim]
      W_s = W_[:,FLAGS.latent_dim:]
      rng, key = random.split(rng)
      rng, key = random.split(rng)
      b[s] = tfd.Normal(loc=jnp.zeros([x_dims[s]]), scale=1.).sample(seed=key)
      rng, key = random.split(rng)
      W[s] = jnp.concatenate([jnp.zeros([x_dims[s], FLAGS.latent_dim * (int(s))]), W_p,
                              jnp.zeros([x_dims[s], FLAGS.latent_dim * (FLAGS.num_modalities - int(s) - 1)]), W_s], 1)


  #sample from standard Gaussian prior
  sigma_z = jnp.ones([latent_dim])  # 1./tf.range(1,1+len(latent_dims), dtype=tf.float32)
  mu_z = jnp.zeros([latent_dim])
  eps = tfd.Normal(loc=jnp.zeros([train_size, latent_dim]), scale=1.).sample(seed=key)
  z = mu_z + sigma_z * eps
  rng, key = random.split(rng)
  x_dict = {}
  for s in modalities:
    x_dict[s] = tfd.Normal(
      loc=jnp.squeeze(jnp.matmul(W[s], jnp.expand_dims(z, -1)), -1)+b[s],
      scale=jnp.sqrt(sigma_squared_x)).sample(seed=key)[:,None]
    rng, key = random.split(rng)


  np.savetxt(os.path.join(output_dir, 'W.csv'), jnp.concatenate([v for v in W.values()], 0))
  np.savetxt(os.path.join(output_dir, 'b.csv'), jnp.concatenate([v for v in b.values()], 0))
  np.savetxt(os.path.join(output_dir, 'sigma_squared_x.csv'), np.array([sigma_squared_x]))

  joint_x = jnp.squeeze(jnp.concatenate([v for k,v in x_dict.items()], -1), 1)
  x_mask = {k: jnp.ones_like(v[:,:,0]) for k,v in x_dict.items()}


  #Compute MLE of training data (we use the fixed noise variance sigma_squared_x instead of its MLE estimate
  #as the fixed value is also used for the decoder model
  joint_b_mle = jnp.mean(joint_x, 0)

  x_centered = joint_x - joint_b_mle
  joint_eigenvalues, joint_eigenvectors = np.linalg.eigh(
    (np.einsum('si,sj->ij', x_centered, x_centered)/x_centered.shape[0]))

  sigma_squared_x_mle = np.mean(joint_eigenvalues[:-latent_dim])
  sigma_squared_x_annealed = sigma_squared_x_mle/FLAGS.beta

  Z_d = jnp.diag(jnp.sqrt(joint_eigenvalues[-latent_dim:] - sigma_squared_x_annealed))
  U_d = (joint_eigenvectors[:,-latent_dim:])
  joint_W_mle = jnp.matmul(U_d, Z_d)

  C_mle = jnp.matmul(joint_W_mle, jnp.transpose(joint_W_mle)) + sigma_squared_x_annealed * jnp.eye(sum([v for v in x_dims.values()]))
  data_dist_mle = tfd.MultivariateNormalTriL(
    loc=joint_b_mle,
    scale_tril=(np.linalg.cholesky(C_mle)))
  data_log_prob_mle = jnp.sum(data_dist_mle.log_prob(joint_x))

  encoders = {}
  decoders = {}

  # encoder model
  class Encoder(nn.Module):
    output_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x, mask):
      x = jnp.where(mask, x, 0.)
      z = nn.Dense(features=self.hidden_dim)(x)
      z = nn.relu(z)
      z = nn.Dense(features=self.hidden_dim)(z)
      z = nn.relu(z)
      z = nn.Dense(features=self.output_dim)(z)
      return z


  # decoder model
  class FixedLinearDecoder(nn.Module):
    W: jnp.array
    b: jnp.array
    scale: float

    @nn.compact
    def __call__(self, z):
      x = jnp.squeeze(jnp.matmul(self.W, jnp.expand_dims(z, -1)), -1)+self.b
      return x, self.scale * jnp.ones_like(x)

  #we fix the decoder variance to the true value which is shared across all views
  class LinearDecoder(nn.Module):
    output_dim : int
    #scale_init: Callable = nn.initializers.constant(0.)
    scale: float
    @nn.compact
    def __call__(self, z):
      x = nn.Dense(features=self.output_dim)(z)
      #scale = self.param('scale', self.scale_init, x.shape[-1]) * jnp.ones_like(x)
      return x, self.scale * jnp.ones_like(x)


  embedding_dim = 2 * latent_dim

  for k in modalities:
    encoders[k] = Encoder(output_dim=embedding_dim, hidden_dim=FLAGS.encoder_dim)
    if not FLAGS.estimate_model:
      decoders[k] = FixedLinearDecoder(W=W[k], b=b[k], scale=.5*jnp.log(sigma_squared_x))
    else:
      decoders[k] = LinearDecoder(output_dim=x_dims[k], scale=.5*jnp.log(sigma_squared_x))

  rng, key = random.split(rng)
  print('build model')
  rec_weights = {k: 1. for k in modalities}

  joint_model = MultiModalVAE(bound=FLAGS.bound,
      decoder_models=decoders, encoder_models=encoders, fixed_prior=True,
      K=1, d=latent_dim, x_types=x_types, x_dims=x_dims,
      T=T, pooling_size=FLAGS.pooling_size, aggregation=FLAGS.aggregation,
      sum_pooling_depth=2, embedding_size=embedding_dim,
      hidden_aggregation_size=FLAGS.aggregation_size,
      attention_size=FLAGS.aggregation_size,
      pooling_aggregation_depth=FLAGS.pooling_aggregation_depth,
      attention_depth=1, act_fn=nn.leaky_relu, beta=FLAGS.beta, rec_weights=rec_weights, stl_grad=True,
      min_encoder_scale=1e-3, min_decoder_scale=1e-3)


  optimizer = optax.chain(
    optax.zero_nans(),
    optax.clip(100.0),
    optax.adam(FLAGS.learning_rate)
  )

  print('model defined')
  state = train_state.TrainState.create(
    apply_fn=joint_model.apply,
    params=joint_model.init(rng, {k:v[:10] for k,v in x_dict.items()},
      { k: v[:10] for k,v in x_mask.items()}, 1, key)['params'],
    tx=optimizer,
  )
  print('model initialised')

  print('start training')
  rng, key = random.split(rng)

  state, grad, loss, rng = joint_model.train(
    state, x_dict, x_mask, key, FLAGS.training_epochs, batch_size=FLAGS.batch_size,
    num_importance_samples=1,
    log_dir=None)
  variational_bound = - loss

  #save model
  checkpoints.save_checkpoint(ckpt_dir=output_dir, target=state, step=state.step, overwrite=True)

  #quantities for model parameters
  joint_W = jnp.concatenate([v for k,v in W.items()], 0)
  joint_b = jnp.concatenate([v for k,v in b.items()], 0)
  K = jnp.matmul(jnp.transpose(joint_W), joint_W) + sigma_squared_x_annealed * jnp.eye(latent_dim)
  K_inv = np.linalg.inv(np.array(K))
  C = jnp.matmul(joint_W, jnp.transpose(joint_W)) + sigma_squared_x_annealed * jnp.eye(sum([v for v in x_dims.values()]))
  data_dist = tfd.MultivariateNormalTriL(
    loc=joint_b,
    scale_tril=(np.linalg.cholesky(C)))
  data_log_prob = jnp.sum(data_dist.log_prob(joint_x))



  #compute posterior distributions given learned generative parameters
  W_dec = {}
  b_dec= {}
  for k in modalities:
    W_dec[k]=jnp.transpose(state.params['decoder_models_'+k]['Dense_0']._dict['kernel'])
    b_dec[k]=jnp.transpose(state.params['decoder_models_'+k]['Dense_0']._dict['bias'])
  joint_W_dec = jnp.concatenate([v for k,v in W_dec.items()], 0)
  joint_b_dec = jnp.concatenate([v for k,v in b_dec.items()], 0)

  C_dec = jnp.matmul(joint_W_dec, jnp.transpose(joint_W_dec)) + sigma_squared_x_annealed * jnp.eye(sum([v for v in x_dims.values()]))
  K_dec = jnp.matmul(jnp.transpose(joint_W_dec), joint_W_dec) + sigma_squared_x_annealed * jnp.eye(latent_dim)
  K_inv_dec = np.linalg.inv(np.array(K_dec))
  joint_posterior_means_dec = jnp.squeeze(
    jnp.matmul(K_inv_dec, jnp.matmul(jnp.transpose(joint_W_dec),jnp.expand_dims(joint_x-joint_b_dec, -1))), -1)
  joint_posterior_cov_dec = jnp.matmul(sigma_squared_x_annealed * jnp.eye(latent_dim), K_inv_dec)
  data_dist_dec = tfd.MultivariateNormalTriL(
    loc=joint_b_dec,
    scale_tril=(np.linalg.cholesky(C_dec)))
  data_log_prob_dec = jnp.sum(data_dist_dec.log_prob(joint_x))

  variational_gap = data_log_prob_dec - variational_bound

  uni_modal_posterior_means = {}
  uni_modal_posterior_cov = {}
  uni_modal_K = {}
  for k in modalities:
    uni_modal_K[k] = jnp.matmul(jnp.transpose(W_dec[k]), W_dec[k]) + sigma_squared_x_annealed * jnp.eye(latent_dim)
    uni_modal_inv_K = np.linalg.inv(np.array(uni_modal_K[k]))
    uni_modal_posterior_means[k] = jnp.squeeze(
      jnp.matmul(uni_modal_inv_K, jnp.matmul(jnp.transpose(W_dec[k]), jnp.expand_dims(x_dict[k][:,0,:] - b_dec[k], -1))), -1)
    uni_modal_posterior_cov[k] = jnp.matmul(sigma_squared_x_annealed * jnp.eye(latent_dim), uni_modal_inv_K)

  #compute variational approximations and information quantities
  encoding_features_multi_modal, recon_log_prob_full, recon_log_prob_marginal, recon_log_prob_cross,\
        log_q_z_partial, log_p_z_partial, log_q_z_full, log_q_z_partial_at_full, log_p_z_full, sampled_masks\
    = joint_model.encode(state, x_dict, x_mask, key, mask_prob=None)
  encoding_features_uni_modal = {}
  uni_modal_kl = {}
  for k in modalities:
    x_mask_unimodal = x_mask.copy()
    x_mask_partial = x_mask.copy()
    for j in modalities:
      if j != k:
        x_mask_unimodal[j] = 0. * x_mask_unimodal[j]
      if j == k:
        x_mask_partial[j] = 0. * x_mask_unimodal[j]
    encoding_features_uni_modal[k], _, _, _, _, _, _, _, _, _ = joint_model.encode(
      state, x_dict, x_mask_unimodal, key)
    uni_modal_posterior = tfd.MultivariateNormalTriL(
      loc=uni_modal_posterior_means[k],
      scale_tril=(np.linalg.cholesky(uni_modal_posterior_cov[k][None])*jnp.ones([train_size,1,1])))
    uni_modal_mean, uni_modal_scale = jnp.split(encoding_features_uni_modal[k], 2, -1)
    if FLAGS.aggregation != 'MoE':
      uni_modal_approximation = tfd.MultivariateNormalDiag(loc=uni_modal_mean[:,0,:],
                                                         scale_diag=jnp.exp(uni_modal_scale[:,0,:]))
    else:
      #Note there are no missing views here, so this becomes just a classic, i.e. uni-modal, Gaussian
      uni_modal_approximation = tfd.MultivariateNormalDiag(loc=uni_modal_mean[:, int(k), 0, :],
                                                           scale_diag=jnp.exp(uni_modal_scale[:, int(k), 0, :]))

    uni_modal_kl[k] = tfd.kl_divergence(uni_modal_approximation, uni_modal_posterior)

  multi_modal_mean, multi_modal_scale = jnp.split(encoding_features_multi_modal, 2, -1)
  multi_modal_posterior = tfd.MultivariateNormalTriL(
    loc=joint_posterior_means_dec,
    scale_tril=(np.linalg.cholesky(joint_posterior_cov_dec[None]) * jnp.ones([train_size, 1, 1])))



  if FLAGS.aggregation != 'MoE':
    multi_modal_approximation = tfd.MultivariateNormalDiag(loc=multi_modal_mean[:,0,:],
                                                         scale_diag=jnp.exp(multi_modal_scale[:,0,:]))
    multi_modal_kl = tfd.kl_divergence(multi_modal_approximation, multi_modal_posterior)

  else:
    #here are no missing views again
    multi_modal_components_distribution = tfd.Independent(
      tfd.Normal(loc=jnp.transpose(multi_modal_mean[:,:FLAGS.num_modalities,0,:], [0, 1, 2]),
                 scale=jnp.transpose(jnp.exp(multi_modal_scale[:,:FLAGS.num_modalities,0,:]), [0, 1, 2])),
      reinterpreted_batch_ndims=1)
    multi_modal_mixture_weights = tfd.Categorical(
      probs=1./FLAGS.num_modalities * jnp.ones(FLAGS.num_modalities))
    multi_modal_approximation = tfd.MixtureSameFamily(
          mixture_distribution=multi_modal_mixture_weights,
          components_distribution=multi_modal_components_distribution,
          reparameterize=True)
    #Monte Carlo approximation of kl divergence
    multi_modal_approximation_samples = multi_modal_approximation.sample(100, seed=key)
    multi_modal_kl = jnp. mean(multi_modal_approximation.log_prob(multi_modal_approximation_samples) - \
                     multi_modal_posterior.log_prob(multi_modal_approximation_samples), 0)


  pd.DataFrame(multi_modal_kl).to_csv(os.path.join(output_dir, 'multi_modal_kl.csv'), index=False)
  pd.DataFrame(uni_modal_kl).to_csv(os.path.join(output_dir, 'uni_modal_kl.csv'), index=False)

  np.savetxt(os.path.join(output_dir,'data_log_prob_mle.txt'),
             np.array([data_log_prob_mle]))
  np.savetxt(os.path.join(output_dir,'variational_bound.txt'),
             np.array([variational_bound]))
  np.savetxt(os.path.join(output_dir,'variational_gap.txt'),
             np.array([variational_gap]))
  np.savetxt(os.path.join(output_dir,'data_log_prob_dec.txt'),
             np.array([data_log_prob_dec]))


  pd.DataFrame(recon_log_prob_cross).transpose().to_csv(
    os.path.join(output_dir, 'recon_log_prob_cross.csv'), index=False)
  pd.DataFrame(recon_log_prob_marginal).transpose().to_csv(
    os.path.join(output_dir, 'recon_log_prob_marginal.csv'), index=False)
  pd.DataFrame(recon_log_prob_full).transpose().to_csv(
    os.path.join(output_dir, 'recon_log_prob_full.csv'), index=False)

  pd.DataFrame(sampled_masks[:,:,0]).transpose().to_csv(
    os.path.join(output_dir, 'sampled_masks.csv'), index=False)

  pd.DataFrame(log_q_z_full - log_p_z_full).transpose().to_csv(
    os.path.join(output_dir, 'full_rates.csv'), index=False)
  pd.DataFrame(log_q_z_partial - log_p_z_partial).transpose().to_csv(
    os.path.join(output_dir, 'partial_rates.csv'), index=False)
  pd.DataFrame(log_q_z_full - log_q_z_partial_at_full).transpose().to_csv(
    os.path.join(output_dir, 'cross_rates.csv'), index=False)

  np.savetxt(os.path.join(output_dir,'data_log_prob.txt'),
             np.array([data_log_prob]))

  data_log_prob_dec_is, _ = joint_model.llh_eval(state=state, x=x_dict, x_mask=x_mask,
                                                 batch_size=FLAGS.batch_size, num_importance_samples=500, rng=key)

  np.savetxt(os.path.join(output_dir,'data_log_prob_dec_is.txt'),
              np.array([jnp.sum((data_log_prob_dec_is))]))


if __name__ == '__main__':
  app.run(main)



