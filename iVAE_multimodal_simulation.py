####
#Script for running the identifiable bi-modal model
#Data simulation and model are adapted from Khemakhem et al, Variational Autoencoders and Nonlinear ICA: A Unifying Framework
#https://github.com/ilkhem/iVAE/blob/master/data/data.py
#We follow the missingness mechanism from Zhang et al, CPM-Nets: Cross Partial Multi-View Networks
#https://github.com/hanmenghan/CPM_Nets
###

import numpy as np
from absl import app
from absl import flags
import sys
import jax.numpy as jnp
from jax import random
from multimodal_vae import MultiModalVAE
import optax
import os
from flax import linen as nn
from typing import Callable
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from flax.training import train_state, checkpoints
import mcc
import scipy
from itertools import combinations
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


FLAGS = flags.FLAGS
flags.DEFINE_string(
  'bound', default='masked',
  help=('bound type (masked, mixture, tc')
)
flags.DEFINE_float(
    'missing_rate', default=0.5,
    help=('missing rate')
)
flags.DEFINE_integer(
    'seed', default=2,
    help=('seed for rng')
)
flags.DEFINE_integer(
    'latent_dim', default=10,
    help=('dimension of latent state')
)
flags.DEFINE_integer(
    'num_modalities', default=4,
    help=('number of modalities')
)
flags.DEFINE_integer(
    'x_dims', default=25,
    help=('dimension of observations')
)
flags.DEFINE_integer(
    'train_size', default=5000,
    help=('train size for each modality')
)
flags.DEFINE_integer(
    'test_size', default=5000,
    help=('train size for each modality')
)
flags.DEFINE_integer(
    'batch_size', default=250,
    help=('batch size')
)
flags.DEFINE_integer(
    'training_epochs', default=4000,
    help=('training epochs')
)
flags.DEFINE_float(
    'learning_rate', default=.0005,
    help=('learning rate')
)
flags.DEFINE_float(
    'beta', default=1.,
    help=('beta for scaling rate terms')
)
flags.DEFINE_float(
    'scale', default=.5,
    help=('observation scale')
)
flags.DEFINE_string(
    'aggregation', default='SumPoolingMixture',
    help=('multi modal aggregation scheme (PoE, SumPooling, Attention.')
)
flags.DEFINE_string(
  'output_dir', default=os.getcwd(),
  help=('output directory for results,')
)
flags.DEFINE_integer(
  'K_model', default=5,
  help=('number of mixtures for the estimated model,')
)
flags.DEFINE_integer(
  'K_sim', default=5,
  help=('number of clusters in simulated data,')
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
  'encoder_size', default=256,
  help=('MLP hidden dimension of marginal encoders')
)
flags.DEFINE_integer(
  'embedding_dim', default=256,
  help=('embedding dim for aggregation (if not MoE or PoE)')
)
flags.DEFINE_integer(
  'pooling_aggregation_depth', default=2,
  help=('hidden layer depth for pooling nn')
)
flags.DEFINE_string(
  'family', default='Gaussian',
  help=('prior/encoding family Gaussian or Laplace')
)

def main(argv):
  del argv  # Unused.


  FLAGS(sys.argv)
  output_dir = os.path.join(
      FLAGS.output_dir, FLAGS.family, str(FLAGS.latent_dim), str(FLAGS.num_modalities),
      str(FLAGS.beta), str(FLAGS.missing_rate), str(FLAGS.K_model),
      str(FLAGS.bound), str(FLAGS.aggregation), str(FLAGS.encoder_size), str(FLAGS.seed))


  if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

  fv = flags._flagvalues.FlagValues()
  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  flag_file = open(os.path.join(output_dir, 'flags.txt'), "w")
  flag_file.write(s)
  flag_file.close()


  #######
  #Copied from code for iVAE paper
  #######
  def to_one_hot(x, m=None):
    if type(x) is not list:
      x = [x]
    if m is None:
      ml = []
      for xi in x:
        ml += [xi.max() + 1]
      m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
      xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
      xoh[i][np.arange(xi.size), xi.astype(int)] = 1
    return xoh


  def lrelu(x, neg_slope):
    """
    Leaky ReLU activation function
    @param x: input array
    @param neg_slope: slope for negative values
    @return:
        out: output rectified array
    """

    def _lrelu_1d(_x, _neg_slope):
      """
      one dimensional implementation of leaky ReLU
      """
      if _x > 0:
        return _x
      else:
        return _x * _neg_slope

    leaky1d = np.vectorize(_lrelu_1d)
    assert neg_slope > 0  # must be positive
    return leaky1d(x, neg_slope)


  def sigmoid(x):
    """
    Sigmoid activation function
    @param x: input array
    @return:
        out: output array
    """
    return 1 / (1 + np.exp(-x))


  def generate_mixing_matrix(d_sources: int, d_data=None, lin_type='uniform', cond_threshold=25, n_iter_4_cond=None,
                             dtype=np.float32):
    """
    Generate square linear mixing matrix
    @param d_sources: dimension of the latent sources
    @param d_data: dimension of the mixed data
    @param lin_type: specifies the type of matrix entries; either `uniform` or `orthogonal`.
    @param cond_threshold: higher bound on the condition number of the matrix to ensure well-conditioned problem
    @param n_iter_4_cond: or instead, number of iteration to compute condition threshold of the mixing matrix.
        cond_threshold is ignored in this case/
    @param dtype: data type for data
    @return:
        A: mixing matrix
    @rtype: np.ndarray
    """
    if d_data is None:
      d_data = d_sources

    if lin_type == 'orthogonal':
      A = (np.linalg.qr(np.random.uniform(-1, 1, (d_sources, d_data)))[0]).astype(dtype)

    elif lin_type == 'uniform':
      if n_iter_4_cond is None:
        cond_thresh = cond_threshold
      else:
        cond_list = []
        for _ in range(int(n_iter_4_cond)):
          A = np.random.uniform(-1, 1, (d_sources, d_data)).astype(dtype)
          for i in range(d_data):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
          cond_list.append(np.linalg.cond(A))

        cond_thresh = np.percentile(cond_list, 25)  # only accept those below 25% percentile

      A = (np.random.uniform(0, 2, (d_sources, d_data)) - 1).astype(dtype)
      for i in range(d_data):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

      while np.linalg.cond(A) > cond_thresh:
        # generate a new A matrix!
        A = (np.random.uniform(0, 2, (d_sources, d_data)) - 1).astype(dtype)
        for i in range(d_data):
          A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
    else:
      raise ValueError('incorrect method')
    return A




  def generate_nonstationary_sources(n_per_seg: int, n_seg: int, d: int, prior='lap', var_bounds=np.array([0.5, 3]),
                                     dtype=np.float32, uncentered=False):
    """
    Generate source signal following a TCL distribution. Within each segment, sources are independent.
    The distribution withing each segment is given by the keyword `dist`
    @param n_per_seg: number of points per segment
    @param n_seg: number of segments
    @param d: dimension of the sources same as data
    @param prior: distribution of the sources. can be `lap` for Laplace , `hs` for Hypersecant or `gauss` for Gaussian
    @param var_bounds: optional, upper and lower bounds for the modulation parameter
    @param dtype: data type for data
    @param bool uncentered: True to generate uncentered data
    @return:
        sources: output source array of shape (n, d)
        labels: label for each point; the label is the component
        m: mean of each component
        L: modulation parameter of each component
    @rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    var_lb = var_bounds[0]
    var_ub = var_bounds[1]
    n = n_per_seg * n_seg

    L = np.random.uniform(var_lb, var_ub, (n_seg, d))
    if uncentered:
      m = np.random.uniform(-5, 5, (n_seg, d))
    else:
      m = np.zeros((n_seg, d))

    labels = np.zeros(n, dtype=dtype)
    if prior == 'lap':
      sources = np.random.laplace(0, 1 / np.sqrt(2), (n, d)).astype(dtype)
    elif prior == 'hs':
      sources = scipy.stats.hypsecant.rvs(0, 1, (n, d)).astype(dtype)
    elif prior == 'gauss':
      sources = np.random.randn(n, d).astype(dtype)
    else:
      raise ValueError('incorrect dist')

    for seg in range(n_seg):
      segID = range(n_per_seg * seg, n_per_seg * (seg + 1))
      sources[segID] *= L[seg]
      sources[segID] += m[seg]
      labels[segID] = seg

    return sources, labels, m, L

  def generate_multimodal_data(n_per_seg, n_seg, d_sources, d_data=None, n_layers=3, prior='lap', activation='lrelu',
                    batch_size=250,
                    seed=10, slope=.1, var_bounds=np.array([0.5, 3]), lin_type='uniform', n_iter_4_cond=1e4,
                    dtype=np.float32, uncentered=False, noisy=0, n_modalities=1):
    """
    Generate artificial data with arbitrary mixing
    @param int n_per_seg: number of observations per segment
    @param int n_seg: number of segments
    @param int d_sources: dimension of the latent sources
    @param int or None d_data: dimension of the data
    @param int n_layers: number of layers in the mixing MLP
    @param str activation: activation function for the mixing MLP; can be `none, `lrelu`, `xtanh` or `sigmoid`
    @param str prior: prior distribution of the sources; can be `lap` for Laplace or `hs` for Hypersecant
    @param int batch_size: batch size if data is to be returned as batches. 0 for a single batch of size n
    @param int seed: random seed
    @param var_bounds: upper and lower bounds for the modulation parameter
    @param float slope: slope parameter for `lrelu` or `xtanh`
    @param str lin_type: specifies the type of matrix entries; can be `uniform` or `orthogonal`
    @param int n_iter_4_cond: number of iteration to compute condition threshold of the mixing matrix
    @param dtype: data type for data
    @param bool uncentered: True to generate uncentered data
    @param float noisy: if non-zero, controls the level of noise added to observations
    @param int n_modalities: number of modalities

    @return:
        tuple of batches of generated (sources, data, auxiliary variables, mean, variance)
    @rtype: tuple
    """
    if seed is not None:
      np.random.seed(seed)

    if d_data is None:
      d_data = d_sources

    # sources
    sources, labels, m, L = generate_nonstationary_sources(n_per_seg, n_seg, d_sources, prior=prior,
                                                           var_bounds=var_bounds,
                                                           dtype=dtype, uncentered=uncentered)
    n = n_per_seg * n_seg

    # non linearity
    if activation == 'lrelu':
      act_f = lambda x: lrelu(x, slope).astype(dtype)
    elif activation == 'sigmoid':
      act_f = sigmoid
    elif activation == 'xtanh':
      act_f = lambda x: np.tanh(x) + slope * x
    elif activation == 'none':
      act_f = lambda x: x
    else:
      raise ValueError('incorrect non linearity: {}'.format(activation))

    Xs= {}
    for m in range(n_modalities):
      # Mixing time!
      assert n_layers > 1  # suppose we always have at least 2 layers. The last layer doesn't have a non-linearity
      A = generate_mixing_matrix(d_sources, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
      X = act_f(np.dot(sources, A))
      if d_sources != d_data:
        B = generate_mixing_matrix(d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
      else:
        B = A
      for nl in range(1, n_layers):
        if nl == n_layers - 1:
          X = np.dot(X, B)
        else:
          X = act_f(np.dot(X, B))

      # add noise:
      if noisy:
        X += noisy * np.random.randn(*X.shape)
      Xs[str(m)]=(X)
    # always return batches (as a list), even if number of batches is one,
    if not batch_size:
      return [sources], [Xs], to_one_hot([labels], m=n_seg), m, L
    else:
      idx = np.random.permutation(n)
      Xb, Sb, Ub = [], [], []
      n_batches = int(n / batch_size)
      for c in range(n_batches):
        Sb += [sources[idx][c * batch_size:(c + 1) * batch_size]]
        Xb += [X[idx][c * batch_size:(c + 1) * batch_size]]
        Ub += [labels[idx][c * batch_size:(c + 1) * batch_size]]
      return Sb, Xb, to_one_hot(Ub, m=n_seg), m, L


  #######

  rng = random.PRNGKey(FLAGS.seed)
  rng, key = random.split(rng)
  #simulate data generating parameters
  modalities = [str(k) for k in range(0,FLAGS.num_modalities)]
  latent_dim = FLAGS.latent_dim
  T = 1


  rng, key = random.split(rng)

  Sb, Xb, Ub, m, L = generate_multimodal_data(
    n_per_seg=(FLAGS.train_size+FLAGS.test_size)//FLAGS.K_sim, n_seg=FLAGS.K_sim, d_sources=FLAGS.latent_dim,
    d_data=FLAGS.x_dims, prior='gauss' if FLAGS.family == 'Gaussian' else 'lap', seed=FLAGS.seed, batch_size=None,
    noisy=FLAGS.scale, uncentered=True, n_modalities=FLAGS.num_modalities)
  Sb, Xb, Ub = Sb[0], Xb[0], Ub[0]
  np.savez_compressed(os.path.join(output_dir, 'data') , s=Sb, x=Xb, u=Ub, m=m, L=L)

  ###
  #Copied from CPM net paper (https://github.com/hanmenghan/CPM_Nets)
  ###
  def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return:Sn
    """
    one_rate = 1 - missing_rate
    if one_rate <= (1 / view_num):
      enc = OneHotEncoder()
      view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
      return view_preserve
    error = 1
    if one_rate == 1:
      matrix = randint(1, 2, size=(alldata_len, view_num))
      return matrix
    while error >= 0.005:
      enc = OneHotEncoder()
      view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
      one_num = view_num * alldata_len * one_rate - alldata_len
      ratio = one_num / (view_num * alldata_len)
      matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
      a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
      one_num_iter = one_num / (1 - a / one_num)
      ratio = one_num_iter / (view_num * alldata_len)
      matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
      matrix = ((matrix_iter + view_preserve) > 0).astype(int)
      ratio = np.sum(matrix) / (view_num * alldata_len)
      error = abs(one_rate - ratio)
    return matrix


  view_num = FLAGS.num_modalities
  # Randomly generated missing matrix
  Sn_train_x = get_sn(view_num, FLAGS.train_size, FLAGS.missing_rate)
  #Sn_train = Sn[np.arange(trainData.num_examples)]
  Sn_test_x = get_sn(view_num, FLAGS.test_size, FLAGS.missing_rate)
  np.savez_compressed(os.path.join(output_dir, 'misssing_masks') , Sn_train_x=Sn_train_x, Sn_test_x=Sn_test_x)


  x_dict_train = {k: Xb[k][:FLAGS.train_size,None,:] for k in modalities}
  x_dict_test = {k: Xb[k][FLAGS.train_size:,None,:] for k in modalities}
  x_mask_train = {k: Sn_train_x[:,int(k)][:,None] for k in modalities}
  x_mask_test = {k:  Sn_test_x[:,int(k)][:,None] for k in modalities}

  encoders = {}
  decoders = {}

  # encoder model
  class MLPEncoder(nn.Module):
    output_dim: int
    layer_size: list

    @nn.compact
    def __call__(self, x, mask):
      z = jnp.where(mask, x, 0.)
      for l in range(len(self.layer_size)):
        z = nn.Dense(features=self.layer_size[l])(z)
        z = nn.relu(z)
      z = nn.Dense(features=self.output_dim)(z)
      print(z.shape)
      return z


  # decoder model
  class InjectiveMLPDecoder(nn.Module):
    output_dim: int
    input_dim: int
    use_scale: bool
    scale_init: Callable = nn.initializers.constant(0.)


    @nn.compact
    def __call__(self, z):
      y = nn.Dense(features=(self.output_dim))(z)
      y = nn.leaky_relu(y, negative_slope=.1)
      x = nn.Dense(features=self.output_dim)(y)
      scale = self.param('scale', self.scale_init, x.shape[-1]) * jnp.ones_like(x)
      if self.use_scale:
        return x, scale
      else:
        return jnp.expand_dims(x, -2)


  embedding_dim = 2 * latent_dim if FLAGS.aggregation in ['MoE', 'PoE'] else FLAGS.aggregation_size
  encoder_layer_sizes = [FLAGS.encoder_size, FLAGS.encoder_size]

  for k in modalities:
    encoders[k] = MLPEncoder(output_dim=embedding_dim, layer_size=encoder_layer_sizes)
    decoders[k] = InjectiveMLPDecoder(output_dim=FLAGS.x_dims, input_dim=latent_dim, use_scale=True)

  x_dims = {k:  FLAGS.x_dims for k in modalities}
  x_types = {k: 'normal' for k in modalities}

  rng, key = random.split(rng)
  print('build model')
  fixed_prior = True if FLAGS.K_model == 1 else False
  rec_weights = {k: 1. for k in modalities}

  joint_model = MultiModalVAE(bound=FLAGS.bound,
      decoder_models=decoders, encoder_models=encoders, fixed_prior=fixed_prior, prior=FLAGS.family,
      K=FLAGS.K_model, d=latent_dim, x_types=x_types, x_dims=x_dims, posterior=FLAGS.family,
      T=T, pooling_size=FLAGS.pooling_size, aggregation=FLAGS.aggregation, beta=FLAGS.beta,
      sum_pooling_depth=2, num_heads=4, embedding_size=embedding_dim,
      hidden_aggregation_size=FLAGS.aggregation_size,
      attention_size=FLAGS.aggregation_size,
      pooling_aggregation_depth=FLAGS.pooling_aggregation_depth,
      attention_depth=1, pre_layer_norm=True,
      act_fn=nn.leaky_relu, stl_grad=True, rec_weights=rec_weights)

  print('model defined')

  optimizer = optax.chain(
    optax.zero_nans(),
    optax.clip(1000.0),
    optax.adam(FLAGS.learning_rate)
  )

  state = train_state.TrainState.create(
    apply_fn=joint_model.apply,
    params=joint_model.init(rng, {k:v[:10] for k,v in x_dict_train.items()},
      { k: v[:10] for k,v in x_mask_train.items()}, 1, key)['params'],
    tx=optimizer,
  )
  print('model initialised')

  print('start training')
  rng, key = random.split(rng)


  state, grad, loss, rng = joint_model.train(
    state, x_dict_train, x_mask_train, key, FLAGS.training_epochs, batch_size=FLAGS.batch_size,
    num_importance_samples=1)
  variational_bound = - loss

  #save model
  checkpoints.save_checkpoint(ckpt_dir=output_dir, target=state, step=state.step, overwrite=True)

  # CCA to align latent representations and ground truth
  x_full, x_prior, z_full, log_q_c, perms, rng = joint_model.generate(
    state=state, x=x_dict_test, x_mask_cond=x_mask_test, rng=key, batch_size=FLAGS.batch_size)

  data_log_prob_is, _ = joint_model.llh_eval(state=state, x=x_dict_test, x_mask=x_mask_test,
                                             batch_size=FLAGS.batch_size, num_importance_samples=512, rng=key)
  np.savetxt(os.path.join(output_dir,'data_log_prob_is.txt'),
              np.array([jnp.sum((data_log_prob_is))]))
  np.savetxt(os.path.join(output_dir, 'loss.txt'),
             np.array([loss]))

  data_log_prob_is_train, _ = joint_model.llh_eval(state=state, x=x_dict_train, x_mask=x_mask_train,
                                             batch_size=FLAGS.batch_size, num_importance_samples=512, rng=key)
  np.savetxt(os.path.join(output_dir,'data_log_prob_is_train.txt'),
              np.array([jnp.sum((data_log_prob_is_train))]))

  z_test = Sb[FLAGS.train_size:]
  u_test = Ub[FLAGS.train_size:]
  mcc_weak_in, mcc_weak_out, mcc_strong_in, mcc_strong_out = mcc.compute_mcc(
      np.array(z_test[perms.reshape([-1])]), np.array(z_full[0, :, 0, :]), cca_dim=1)
  print(mcc_weak_in, mcc_weak_out, mcc_strong_in, mcc_strong_out)
  np.savetxt(os.path.join(output_dir, 'mcc_weak_in.txt'),
             np.array([mcc_weak_in]))
  np.savetxt(os.path.join(output_dir, 'mcc_weak_out.txt'),
             np.array([mcc_weak_out]))
  np.savetxt(os.path.join(output_dir, 'mcc_strong_in.txt'),
             np.array([mcc_strong_in]))
  np.savetxt(os.path.join(output_dir, 'mcc_strong_out.txt'),
             np.array([mcc_strong_out]))

  for k1,k2 in combinations(modalities,2):
    x_mask_partial = x_mask_test.copy()
    for j in modalities:
      if j not in [k1,k2]:
        x_mask_partial[j] = 0. * x_mask_partial[j]

    x_partial, x_prior, z_partial, log_q_c, perms, rng = joint_model.generate(
      state=state, x=x_dict_test, x_mask_cond=x_mask_partial, rng=key, batch_size=FLAGS.batch_size)

    z_test = Sb[FLAGS.train_size:]
    u_test = Ub[FLAGS.train_size:]
    mcc_weak_in, mcc_weak_out, mcc_strong_in, mcc_strong_out = mcc.compute_mcc(
        np.array(z_test[perms.reshape([-1])]), np.array(z_partial[0, :, 0, :]), cca_dim=1)
    print(mcc_weak_in, mcc_weak_out, mcc_strong_in, mcc_strong_out)
    np.savetxt(os.path.join(output_dir, 'mcc_weak_in_{}_{}.txt'.format(k1,k2)),
               np.array([mcc_weak_in]))
    np.savetxt(os.path.join(output_dir, 'mcc_weak_out_{}_{}.txt').format(k1,k2),
               np.array([mcc_weak_out]))
    np.savetxt(os.path.join(output_dir, 'mcc_strong_in_{}_{}.txt'.format(k1,k2)),
               np.array([mcc_strong_in]))
    np.savetxt(os.path.join(output_dir, 'mcc_strong_out_{}_{}.txt'.format(k1,k2)),
               np.array([mcc_strong_out]))




if __name__ == '__main__':
  app.run(main)
