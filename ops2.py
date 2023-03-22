import math
import numpy as np
import tensorflow as tf
#import gauss
from scipy import signal
from scipy import ndimage

from tensorflow.python.framework import ops

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
          self.epsilon  = epsilon
          self.momentum = momentum
          self.name = name
    
    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                          decay=self.momentum, 
                          updates_collections=None,
                          epsilon=self.epsilon,
                          scale=True,
                          is_training=train,
                          scope=self.name)

def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-((logits + eps) * tf.log(targets + eps) +
                              (1. - logits +eps) * tf.log(1. - targets + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
    
def dilated_conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=1, d_w=1, dilation_rate=1, stddev=0.02,
           name="d_conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.atrous_conv2d(input_, w, rate = dilation_rate, padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv
        
def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1] ,padding='SAME')

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
        
def dilated_deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=1, d_w=1, dilation_rate=1, stddev=0.02,
             name="d_deconv2d", padding = 'SAME', with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        d_deconv = tf.nn.atrous_conv2d_transpose(input_, w, output_shape=output_shape,rate=dilation_rate,padding=padding)
        
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def deconv(input_, w, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2,
             name="deconv"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]      
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])
        return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear_ex(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, with_bias = True):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        if with_bias is True:
            bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        else:
            bias = tf.zeros([shape[0], output_size], tf.float32) 
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))

        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def fully_connected(input_, output_size, scope=None, stddev=0.1, with_bias = True):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "FC"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
            tf.random_normal_initializer(stddev=stddev))

        result = tf.matmul(input_, matrix)

        if with_bias:
            bias = tf.get_variable("bias", [1, output_size],
                initializer=tf.random_normal_initializer(stddev=stddev))
            result += bias*tf.ones([shape[0], 1], dtype=tf.float32)

        return result


def batch_normal(x , scope="batch_normal" , reuse=False):
    shape = x.get_shape().as_list()
    x_shape = x.get_shape()
    axis = list(range(len(x_shape) - 1))

    with tf.variable_scope(scope) as scope:
        gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))

        mean, variance = tf.nn.moments(x, axis)

        return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, beta, gamma, 0.00001,
                scale_after_normalization=True)
        
def mean_sigmoid_cross_entropy_with_logits(logit, truth):
            '''
            truth: 0. or 1.
            '''
            return tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logit,
                    truth * tf.ones_like(logit)))


def prelu(_x, i):
    alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

def psnr_imgs(img1, img2):
    img1 *= 1.0/img1.max()
    img2 *= 1.0/img2.max()
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    PIXEL_MAX = 1.0
    return 20*math.log10(PIXEL_MAX/rmse)

def compute_psnr(ref, target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    PIXEL_MAX = 1.0
    psnr = 10. * (tf.log(PIXEL_MAX * PIXEL_MAX / mse) / tf.log(10.))

    return psnr

EPSILON = 1e-6

def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    c = np.log(2 * np.pi)
    var = tf.exp(log_var)
    x_mu2 = tf.square(tf.sub(x, mu))   # [Issue] not sure the dim works or not?
    x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = tf.reduce_sum(log_prob, -1, name=name)   # keep_dims=True,
    return log_prob


def GaussianKLD(mu1, lv1, mu2, lv2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        lv: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
    with tf.name_scope('GaussianKLD'):
        v1 = tf.exp(lv1)
        v2 = tf.exp(lv2)
        mu_diff_sq = tf.square(mu1 - mu2)
        dimwise_kld = .5 * (
            (lv2 - lv1) + tf.div(v1, v2) + tf.div(mu_diff_sq, v2) - 1.)
        return tf.reduce_sum(dimwise_kld, -1)
# =============================================================================
# def ssim(img1, img2, cs_map=False):
#     """Return the Structural Similarity Map corresponding to input images img1 
#     and img2 (images are assumed to be uint8)
#     
#     This function attempts to mimic precisely the functionality of ssim.m a 
#     MATLAB provided by the author's of SSIM
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
#     """
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     size = 11
#     sigma = 1.5
#     window = gauss.fspecial_gauss(size, sigma)
#     K1 = 0.01
#     K2 = 0.03
#     L = 255 #bitdepth of image
#     C1 = (K1*L)**2
#     C2 = (K2*L)**2
#     mu1 = signal.fftconvolve(window, img1, mode='valid')
#     mu2 = signal.fftconvolve(window, img2, mode='valid')
#     mu1_sq = mu1*mu1
#     mu2_sq = mu2*mu2
#     mu1_mu2 = mu1*mu2
#     sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
#     sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
#     sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
#     if cs_map:
#         return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
#                     (sigma1_sq + sigma2_sq + C2)), 
#                 (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
#     else:
#         return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
#                     (sigma1_sq + sigma2_sq + C2))
# =============================================================================
