# For Compute Canada submission
# Citation:
# Benjamin Shi (Bin Shi)
# April 18, 2020
#
# variational autoencoder codes: the template of the codes are from GitHub
# https://github.com/altosaar/variational-autoencoder
# make a lot of changes on the params, net structures. And the original codes output Bernoulli distribution,
# now we changed it to Gaussian for handling grey SEM images.
# Two different datasets are run: MPimgallspecial.npy and MPimgallcircle.npy
import itertools
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as tfk
import time
import tensorflow_probability as tfp
from imageio import imwrite

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tfkl = tfk.layers
tfc = tf.compat.v1

nb_filters1 = 64
kernel_size1 = (5, 5)
strides1 = (7, 7)
nb_filters2 = 128
kernel_size2 = (5, 5)
strides2 = (2, 2)
pool_size1 = (2, 2)

nb_filters3 = 256
kernel_size3 = (3, 3)
strides3 = (2, 2)
nb_filters4 = 512
kernel_size4 = (3, 3)
strides4 = (1, 1)
pool_size2 = (5, 5)

flags = tf.app.flags
# flags.DEFINE_string('data_dir', '../../data/', 'Directory for data')
# flags.DEFINE_string('logdir', '/home/benjamin/Data/log/', 'Directory for logs')
flags.DEFINE_string('data_dir', '/home/shibin2/projects/def-ajbonner/shibin2/data', 'Directory for data')
flags.DEFINE_string('logdir', '/home/shibin2/projects/def-ajbonner/shibin2/log/log2', 'Directory for logs')
flags.DEFINE_integer('latent_dim', 500, 'Latent dimensionality of model')
flags.DEFINE_integer('batch_size', 32, 'Minibatch size')
flags.DEFINE_integer('n_samples', 1, 'Number of samples to save')
flags.DEFINE_integer('print_every', 2000, 'Print every n iterations')
flags.DEFINE_integer('hidden_size', 4000, 'Hidden size for neural networks')
flags.DEFINE_integer('n_iterations', 100000, 'number of iterations')

FLAGS = flags.FLAGS


def inference_network(x, latent_dim, hidden_size):
    """Construct an inference network parametrizing a Gaussian.

  Args:
    x: A batch of MNIST digits.
    latent_dim: The latent dimensionality.
    hidden_size: The size of the neural net hidden layers.

  Returns:
    mu: Mean parameters for the variational family Normal
    sigma: Standard deviation parameters for the variational family Normal
  """
    #  ##### we can add some dropout later on
    inference_net = tfk.Sequential([
        tfkl.Convolution2D(nb_filters1, (7, 7), (5, 5), padding='same', activation=tf.nn.relu),
        tfkl.Convolution2D(nb_filters1, (5, 5), (1, 1), padding='same', activation=tf.nn.relu),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Convolution2D(nb_filters2, (3, 3), (1, 1), padding='same', activation=tf.nn.relu),
        tfkl.Convolution2D(nb_filters2, (3, 3), (1, 1), padding='same', activation=tf.nn.relu),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Convolution2D(nb_filters3, kernel_size3, strides3, padding='same', activation=tf.nn.relu),
        tfkl.Convolution2D(nb_filters4, kernel_size4, strides4, padding='same', activation=tf.nn.relu),
        tfkl.MaxPooling2D(pool_size=pool_size2),
        tfkl.Flatten(),
        tfkl.Dense(hidden_size, activation=tf.nn.relu),
        tfkl.Dense(hidden_size, activation=tf.nn.relu),
        tfkl.Dense(latent_dim * 2, activation=None)
    ])
    gaussian_params = inference_net(x)
    # The mean parameter is unconstrained
    mu = gaussian_params[:, :latent_dim]
    # The standard deviation must be positive. Parametrize with a softplus
    sigma = tf.nn.softplus(gaussian_params[:, latent_dim:])
    return mu, sigma


def generative_network(z1, z2, hidden_size):
    """Build a generative network parametrizing the likelihood of the data

  Args:
    z1: Samples of latent variables
    z2: Samples of latent variables2
    hidden_size: Size of the hidden state of the neural net

  Returns:
    bernoulli_logits: logits for the Bernoulli likelihood of the data
  """
    generative_net = tfk.Sequential([
        tfkl.Dense(hidden_size, activation=tf.nn.relu),
        tfkl.Dense(512 * 3 * 5, activation=tf.nn.relu),
        tfkl.Reshape((3, 5, 512)),
        tfkl.UpSampling2D(size=(5, 5)),
        tfkl.Convolution2DTranspose(nb_filters3, kernel_size4, strides4, padding='same', activation=tf.nn.relu),
        tfkl.Convolution2DTranspose(nb_filters2, kernel_size3, strides3, padding='same', activation=tf.nn.relu),
        tfkl.UpSampling2D(size=(2, 2)),
        tfkl.Convolution2D(nb_filters2, (3, 3), (1, 1), padding='same', activation=tf.nn.relu),
        tfkl.Convolution2D(nb_filters1, (3, 3), (1, 1), padding='same', activation=tf.nn.relu),
        tfkl.UpSampling2D(size=(2, 2)),
        tfkl.Convolution2DTranspose(nb_filters1, (5, 5), (1, 1), padding='same', activation=tf.nn.relu),
        tfkl.Convolution2DTranspose(1, (7, 7), (5, 5), padding='same', activation=tfk.activations.sigmoid)
    ])
    normal_logits = generative_net(z1)
    normal_logits2 = generative_net(z2)
    return [tf.reshape(normal_logits, [-1, 600, 1000, 1]), tf.reshape(normal_logits2, [-1, 600, 1000, 1])]


def train():
    # Train a Variational Autoencoder on MNIST

    # Input placeholders
    with tf.name_scope('data'):
        x = tfc.placeholder(tf.float32, [None, 600, 1000, 1])
        # tfc.summary.image('data', x)

    with tfc.variable_scope('variational'):
        q_mu, q_sigma = inference_network(x=x,
                                          latent_dim=FLAGS.latent_dim,
                                          hidden_size=FLAGS.hidden_size)
        # The variational distribution is a Normal with mean and standard
        # deviation given by the inference network
        q_z = tfp.distributions.Normal(loc=q_mu, scale=q_sigma)
        assert q_z.reparameterization_type == tfp.distributions.FULLY_REPARAMETERIZED

    with tfc.variable_scope('model'):
        p_z = tfp.distributions.Normal(loc=np.zeros(FLAGS.latent_dim, dtype=np.float32),
                                       scale=np.ones(FLAGS.latent_dim, dtype=np.float32))
        p_z2 = tfp.distributions.Normal(loc=np.zeros((FLAGS.batch_size, FLAGS.latent_dim), dtype=np.float32),
                                        scale=np.ones((FLAGS.batch_size, FLAGS.latent_dim), dtype=np.float32))
        # The likelihood is Bernoulli-distributed with logits given by the
        # generative network
        [p_x_given_z_logits, p_x_given_z_logits2] = generative_network(z1=q_z.sample(), z2=p_z2.sample(),
                                                                       hidden_size=FLAGS.hidden_size)
        p_x_given_z = tfp.distributions.Normal(loc=p_x_given_z_logits, scale=tf.ones_like(p_x_given_z_logits,
                                                                                          dtype=np.float32))
        # recovered_data = p_x_given_z_logits * 255
        recovered_data = p_x_given_z_logits
        # tfc.summary.image('recovered_data',
        #                   tf.cast(recovered_data, tf.float32))
        # generated
        generated_data = p_x_given_z_logits2
        # tfc.summary.image('posterior_predictive',
        #                   tf.cast(posterior_predictive_samples, tf.float32))

    # Take samples from the prior
    '''
    with tfc.variable_scope('model', reuse=True):
        p_z = tfp.distributions.Normal(loc=np.zeros(FLAGS.latent_dim, dtype=np.float32),
                                       scale=np.ones(FLAGS.latent_dim, dtype=np.float32))
    '''
    '''
        p_z_sample = p_z.sample(FLAGS.n_samples)
        p_x_given_z_logits2 = generative_network(z=p_z_sample,
                                                 hidden_size=FLAGS.hidden_size)
        prior_predictive = tfp.distributions.Normal(loc=p_x_given_z_logits2, scale=tf.ones_like(p_x_given_z_logits2,
                                                                                                dtype=np.float32))
        prior_predictive_samples = prior_predictive.sample()
        prior_predictive_samples = tfk.activations.sigmoid(prior_predictive_samples) * 255
        prior_predictive_samples = tf.round(prior_predictive_samples)
        tfc.summary.image('prior_predictive',
                          tf.cast(prior_predictive_samples, tf.float32))
    '''

    '''
        # Take samples from the prior with a placeholder
        with tfc.variable_scope('model', reuse=True):
            z_input = tf.placeholder(tf.float32, [None, FLAGS.latent_dim])
            p_x_given_z_logits = generative_network(z=z_input,
                                                    hidden_size=FLAGS.hidden_size)
            prior_predictive_inp = tfp.distributions.Bernoulli(logits=p_x_given_z_logits)
            prior_predictive_inp_sample = prior_predictive_inp.sample()
    '''

    # Build the evidence lower bound (ELBO) or the negative loss
    kl = tf.reduce_sum(tfp.distributions.kl_divergence(q_z, p_z), 1)
    expected_log_likelihood = tf.reduce_sum(-0.5 * tf.square(x - p_x_given_z_logits),
                                            [1, 2, 3])

    elbo = tf.reduce_sum(expected_log_likelihood - kl, 0)
    optimizer = tfc.train.RMSPropOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(-elbo)

    # Merge all the summaries
    # summary_op = tfc.summary.merge_all()

    init_op = tfc.global_variables_initializer()

    # Run training
    sess = tfc.InteractiveSession()
    sess.run(init_op)

    trainbm = np.load(os.path.join(FLAGS.data_dir, 'MPimgallspecial.npy'))
    trainbm = np.divide(trainbm, 255.)
    train_data = tf.data.Dataset.from_tensor_slices(trainbm)
    dataset = train_data.repeat().shuffle(buffer_size=1024).batch(FLAGS.batch_size)
    print(type(dataset))

    print('Saving TensorBoard summaries and images to: %s' % FLAGS.logdir)
    # train_writer = tfc.summary.FileWriter(FLAGS.logdir, sess.graph)
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()

    t0 = time.time()
    for i in range(20001):
        batch = sess.run(one_element)
        np_x = np.reshape(batch, [-1, 600, 1000, 1])
        sess.run(train_op, {x: np_x})

        if i % FLAGS.print_every == 0:
            np_elbo = sess.run(elbo, {x: np_x})
            # train_writer.add_summary(summary_str, i)
            print('Iteration: {0:d} ELBO: {1:.3f} s/iter: {2:.3e}'.format(
                i,
                np_elbo / FLAGS.batch_size,
                (time.time() - t0) / FLAGS.print_every))

            # Save samples
            np_generated_data, np_recovered_data = sess.run(
                [generated_data, recovered_data], {x: np_x})
            for k in range(FLAGS.n_samples):
                f_name = os.path.join(
                    FLAGS.logdir, 'iter_%d_original_image_%d_data.jpg' % (i, k))
                imwrite(f_name, np_x[k, :, :, 0])
                f_name = os.path.join(
                    FLAGS.logdir, 'iter_%d_generated_image_%d_sample.jpg' % (i, k))
                imwrite(f_name, np_generated_data[k, :, :, 0])
                f_name = os.path.join(
                    FLAGS.logdir, 'iter_%d_recovered_image_%d.jpg' % (i, k))
                imwrite(f_name, np_recovered_data[k, :, :, 0])

            t0 = time.time()


if __name__ == '__main__':
    train()

