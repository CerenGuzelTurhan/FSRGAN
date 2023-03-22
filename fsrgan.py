# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:19:33 2018

@author: GaziBM
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from ops2 import *
from read_show_ops import *
from mnist_data import *

class FSRGAN(object):
    def __init__(self,x_dim = 64, y_dim = 64,  is_rgb = True, batch_size = 1, learning_rate = 0.001, 
                 beta1 = 0.9, checkpoint_path = './save/srgan_cppn', scale = 5.0, data_dir = None, 
                 sr_ratio = 1, dataset_name = 'sr', is_ycbcr= True,  is_data_aug = True, dataset_index = None,pretrained_model_path=None):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.dim = x_dim
        self.z_dim = 128
        
        self.batch_size = batch_size

        self.learning_rate = learning_rate
              
        self.checkpoint_path = checkpoint_path
        self.scale = scale
        self.data_dir = data_dir
        self.sr_ratio = sr_ratio
        self.keep_prob = 0.95
        self.momentum = 0.9
        self.beta1 = beta1
        self.pretrained_model_path = pretrained_model_path

        self.log_vars = []
        self.c_dim = 1 if not is_rgb else 3 if not is_ycbcr else 1
        self.is_ycbcr = is_ycbcr
        self.is_rgb = is_rgb
        self.is_data_aug = is_data_aug
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.dataset_index = dataset_index
        
        self.m, self.s, self.d = 4, 12, 56
        lambda1, lambda2, lambda3 = 1e-5, 1, 1

        self.get_weights()
       
        self.x_lr = tf.placeholder(tf.float32, shape = [self.batch_size, x_dim,y_dim, self.c_dim])
        self.x_hr = tf.placeholder(tf.float32, shape = [self.batch_size, x_dim*sr_ratio,y_dim*sr_ratio, self.c_dim])            
        
        self.x_tilde = self.generator(self.x_lr,x_dim * sr_ratio, y_dim * sr_ratio)
      
        self.d_x_tilde, self.f_x_tilde, self.d_tilde_logit = self.discriminator(self.x_tilde)
        self.d_x, self.f_x, self.d_logit = self.discriminator(self.x_hr, reuse = True)

        self.ll_loss = tf.reduce_mean(tf.square(self.f_x-self.f_x_tilde))

        self.d_loss_real = tf.reduce_mean(tf.nn.l2_loss(self.d_logit- tf.ones_like(self.d_logit)))
        self.d_loss_tilde = tf.reduce_mean(tf.nn.l2_loss(self.d_tilde_logit - tf.zeros_like(self.d_tilde_logit)))
        self.d_loss = self.d_loss_real + self.d_loss_tilde
        
        #self.g_loss_tilde = binary_cross_entropy_with_logits(tf.ones_like(self.d_x_tilde), self.d_x_tilde)
        g_loss = tf.reduce_mean(tf.nn.l2_loss(self.d_tilde_logit - tf.ones_like(self.d_tilde_logit)))
        self.pixel_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_tilde - self.x_hr), reduction_indices=0), name = 'pixel_loss')
        self.g_loss = lambda1 * g_loss + lambda2* self.ll_loss  + lambda3* self.pixel_loss

        #self.g_loss = self.g_loss_tilde + self.ll_loss
        #self.gan_loss = self.g_loss + self.d_loss
        
        #self.loss = 0.6*self.pixel_loss + 0.4* self.gan_loss
        
        self.log_vars.append(("Generator Loss", self.g_loss))
        self.log_vars.append(("Discriminator Loss", self.d_loss))
        
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
    
        #self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1= beta1) \
         #                 .minimize(self.loss, var_list=self.t_vars)

        #self.opt =  tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss)
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)

    
    def get_weights(self):
        expand_weight, deconv_weight = 'w{}'.format(self.m + 3), 'w{}'.format(self.m + 4)
        self.weights = {
          'w1': tf.Variable(tf.random_normal([5, 5, self.c_dim , self.d], stddev=0.0378, dtype=tf.float32), name='g_w1'),
          'w2': tf.Variable(tf.random_normal([1, 1, self.d, self.s], stddev=0.3536, dtype=tf.float32), name='g_w2'),
          expand_weight: tf.Variable(tf.random_normal([1, 1, self.s, self.d], stddev=0.189, dtype=tf.float32), name='g_expand_weight'),
          deconv_weight: tf.Variable(tf.random_normal([9, 9, self.c_dim, self.d], stddev=0.0001, dtype=tf.float32), name='g_deconv_weight')
        }

        expand_bias, deconv_bias = 'b{}'.format(self.m + 3), 'b{}'.format(self.m + 4)
        self.biases = {
          'b1': tf.Variable(tf.zeros([self.d]), name='g_b1'),
          'b2': tf.Variable(tf.zeros([self.s]), name='g_b2'),
          expand_bias: tf.Variable(tf.zeros([self.d]), name='g_expand_bias'),
          deconv_bias: tf.Variable(tf.zeros([self.c_dim]), name='g_deconv_bias')
        }

        # Create the m mapping layers weights/biases
        for i in range(3, self.m + 3):
          weight_name, bias_name = 'w{}'.format(i), 'b{}'.format(i)
          self.weights[weight_name] = tf.Variable(tf.random_normal([3, 3, self.s, self.s], stddev=0.1179, dtype=tf.float32), name='g_w{}'.format(i))
          self.biases[bias_name] = tf.Variable(tf.zeros([self.s]), name='g_b{}'.format(i))        
   
    def generator(self, img, x_dim = 64, y_dim = 64, reuse = False): # OR DECODER
        with tf.variable_scope('generator') as scope:
            if reuse == True:
                scope.reuse_variables() 
            img = tf.cast(img, tf.float32)   
            self.conv_feature = prelu(tf.nn.conv2d(img, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'], 1)
            # Shrinking
            self.conv_shrink = prelu(tf.nn.conv2d(self.conv_feature, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'], 2)

            # Mapping (# mapping layers = m)
            self.prev_layer, m = self.conv_shrink, self.m
            for i in range(3, m + 3):
                weights, biases = self.weights['w{}'.format(i)], self.biases['b{}'.format(i)]
                self.prev_layer = prelu(tf.nn.conv2d(self.prev_layer, weights, strides=[1,1,1,1], padding='SAME') + biases, i)

            # Expanding
            expand_weights, expand_biases = self.weights['w{}'.format(m + 3)], self.biases['b{}'.format(m + 3)]
            self.conv_expand = prelu(tf.nn.conv2d(self.prev_layer, expand_weights, strides=[1,1,1,1], padding='SAME') + expand_biases, (m + 3))

            # Deconvolution
            deconv_output = [self.batch_size, x_dim, y_dim, self.c_dim]
            deconv_stride = [1,  self.sr_ratio, self.sr_ratio, 1]
            deconv_weights, deconv_biases = self.weights['w{}'.format(m + 4)], self.biases['b{}'.format(m + 4)]
            #conv_deconv = deconv(conv_expand, deconv_weights, deconv_output, 9, 9, self.sr_ratio, self.sr_ratio) + deconv_biases
            self.conv_deconv = tf.nn.conv2d_transpose(self.conv_expand, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME') + deconv_biases

            return self.conv_deconv                                                          
        
    def discriminator(self, img, reuse = False):
        with tf.variable_scope('discriminator') as scope:
            if reuse == True:
                scope.reuse_variables()  
              
            # input part        
            self.h1 = lrelu(conv2d (img, output_dim = 64, k_h = 4, k_w = 4, name='d_h1_conv1'))  
            self.h2 = lrelu(batch_normal(conv2d (self.h1, output_dim = 64, k_h = 4, k_w = 4, name='d_h2_conv2'), scope = 'd_bn1')) 
            self.h3 = lrelu(batch_normal(conv2d (self.h2, output_dim = 64,  k_h = 4, k_w = 4, name='d_h3_conv3'), scope = 'd_bn2')) 
            self.h4 = lrelu(batch_normal(conv2d (self.h3, output_dim = 64,  k_h = 4, k_w = 4,name='d_h4_conv4'), scope='d_bn3')) # 4x4x128
            self.h4 = tf.reshape(self.h4, [self.batch_size, -1])
            self.h5_logit = linear(self.h4, output_size = 1, scope = 'd_h5_lin1')
            self.result = tf.nn.sigmoid(self.h5_logit)
            return self.result, self.h4, self.h5_logit
    def olddiscriminator(self, img, reuse = False):
        with tf.variable_scope('olddiscriminator') as scope:
            if reuse == True:
                scope.reuse_variables()  
            
            d = int (np.ceil(self.x_dim/(2.*2.*2.))) #d means the size of fc layer in discriminator
            # input part        
            self.h1 = lrelu(conv2d (img, output_dim = 32, name='d_h1_conv1'))  
            self.h2 = lrelu(batch_normal(conv2d (self.h1, output_dim = 64, name='d_h2_conv2'), scope = 'd_bn1')) 
            self.h3 = lrelu(batch_normal(conv2d (self.h2, output_dim = 128, name='d_h3_conv3'), scope = 'd_bn2')) 
            h3 = tf.reshape(self.h3, [self.batch_size, -1])
            self.h4 = lrelu(batch_normal(linear (h3, output_size = d*d*128, scope='d_h4_lin1'), scope='d_bn3')) # 4x4x128
            self.h5_logit = linear(self.h4, output_size = 1, scope = 'd_h5_lin2')
            self.result = tf.nn.sigmoid(self.h5_logit)
            return self.result, self.h3, self.h5_logit
        
    def ex_generator(self, img, x_dim = 64, y_dim = 64, reuse = False): # OR DECODER

        with tf.variable_scope('generator') as scope:
            if reuse == True:
                scope.reuse_variables() 
                
            gen_n_points = x_dim * y_dim                
            n_network = 128
            img = tf.cast(img, tf.float32) 
            img_c = tf.reshape(img, [self.batch_size, 1,self.x_dim*self.y_dim*self.c_dim]) * tf.ones([gen_n_points, 1], dtype=tf.float32) 
            img_scaled = img_c * self.scale
            img_unroll = tf.reshape(img_scaled, [self.batch_size*gen_n_points, self.x_dim*self.y_dim*self.c_dim])
# =============================================================================
#             img = tf.convert_to_tensor(img) 
#              _, h, w, c = img.get_shape().as_list()
#             img_c = tf.reshape(img, [self.batch_size, 1,w*h*c]) * tf.ones([gen_n_points*c, 1], dtype=tf.float32) 
#             img_scaled = img_c * self.scale
#             img_unroll = tf.reshape(img_scaled, [self.batch_size*gen_n_points*c, w*h*c])
#             x_unroll = tf.reshape(self.x, [self.batch_size*gen_n_points*c,1])
#             y_unroll = tf.reshape(self.y, [self.batch_size*gen_n_points*c,1])
#             r_unroll = tf.reshape(self.r, [self.batch_size*gen_n_points*c,1])       
# =============================================================================       
            x_unroll = tf.reshape(self.x, [self.batch_size*gen_n_points,1])
            y_unroll = tf.reshape(self.y, [self.batch_size*gen_n_points,1])
            r_unroll = tf.reshape(self.r, [self.batch_size*gen_n_points,1])            
                        
            U = fully_connected(img_unroll, n_network, 'g_0_img') + \
                fully_connected(x_unroll, n_network, 'g_0_x', with_bias = False) + \
                fully_connected(y_unroll, n_network, 'g_0_y', with_bias = False) + \
                fully_connected(r_unroll, n_network, 'g_0_r', with_bias = False)

            H = tf.nn.relu(batch_normal(U, scope = 'g_bn1'))

            for i in range(1, 4):
                H = tf.nn.relu(batch_normal(fully_connected(H, n_network, 'g_h'+str(i)), scope = 'g_bn'+str(i+1)))
            #output = tf.sigmoid(fully_connected(H, 1, 'g_h5'))
            output = tf.sigmoid(fully_connected(H, self.c_dim, 'g_h5'))
        
            result = tf.reshape(output, [self.batch_size, x_dim, y_dim, self.c_dim])
        
            return result  
            
    def generate_sr(self, img, x_dim = 64, y_dim = 64, scale = 5.0):
# =============================================================================
#         if self.is_ycbcr is True:
#             img, cb, cr = rgb2ycbcr(img)
#             img = img[..., np.newaxis]
#         #G = self.generator(img, x_dim, y_dim, reuse = True)
#         
#         G = self.generator(img, x_dim, y_dim, reuse = True)
#         #gen_x, gen_y, gen_r = self.coordinates(x_dim, y_dim, scale = scale)
#         #image = self.sess.run(G, feed_dict={self.x: gen_x, self.y: gen_y, self.r: gen_r})
#         image = self.sess.run(G)
#         if self.is_ycbcr is True:
#             image = ycbcr2rgb(np.reshape(image, image.shape[:-1]), cb, cr)  
# =============================================================================
        model = self.generator(img, x_dim, y_dim, reuse = True)    
        images = self.sess.run(model)
        images = images.reshape(images.shape[:-1])
        return images
          
    def train(self, num_epoch = 100, display_step = 1, checkpoint_step = 10):
         init = tf.initialize_all_variables() 
         self.sess = tf.Session()       
         self.sess.run(init)  
         self.saver = tf.train.Saver()
         old_epoch, epoch = 0, 0
         if self.pretrained_model_path:
             generator_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
             checkpoint_path = self.pretrained_model_path
             ckpt = tf.train.get_checkpoint_state(checkpoint_path)
             if ckpt and ckpt.model_checkpoint_path:
                 ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
             self.sess.run(tf.trainable_variables())
             generator_saver.restore(self.sess, os.path.join(checkpoint_path, ckpt_name))
             self.sess.run(tf.trainable_variables()) 
             ckpt_epoch = int(ckpt_name.split('-')[1])
             old_epoch = ckpt_epoch
          
         ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
         checkpoint_file = os.path.join(self.checkpoint_path , 'model.ckpt')        
         if ckpt:
             self.load_model(self.checkpoint_path)
             print ("Model is loaded!")
             ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
             ckpt_epoch = int(ckpt_name.split('-')[1])
             old_epoch = ckpt_epoch
         ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
         checkpoint_file = os.path.join(self.checkpoint_path , 'model.ckpt')
         old_epoch, epoch = 0, 0
         if ckpt:
             self.load_model(self.checkpoint_path)
             print ("Model is loaded!")
             ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
             ckpt_epoch = int(ckpt_name.split('-')[1])
             old_epoch = ckpt_epoch
         if self.dataset_name == 'mnist':
             images_hr, labels = read_dataset()
             if images_hr.shape[1] != self.x_dim*self.sr_ratio:
                 images_hr = tf.image.resize_images(images_hr, (self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)).eval()
             images_lr = tf.image.resize_images(images_hr, (self.x_dim, self.y_dim)).eval()    
         elif self.dataset_name == 'imagenet':
             images_hr, images_lr, cb, cr =  loadImagenet(self.data_dir, self.x_dim *self.sr_ratio, self.sr_ratio, self.dataset_index, is_ycbcr = True)
         else:
             imgs = read_data(self.data_dir, self.dataset_name)
             images_hr, images_lr, cb, cr  = prepare_data(imgs, size = self.x_dim, sr_ratio = self.sr_ratio, is_rgb = self.is_rgb, is_shuffle = False,  is_data_aug = self.is_data_aug) 
# =============================================================================
#          if self.is_ycbcr is True:
#              rgb_images = images_lr
#              images_lr,images_lr_cb, images_lr_cr = rgb2ycbcr(images_lr)
#              images_lr = images_lr[..., np.newaxis]
#              images_hr,_,_ = rgb2ycbcr(images_hr)
#              images_hr = images_hr[..., np.newaxis]
# =============================================================================
                 
#             if self.is_rgb is False:
#                 images = np.reshape(images, (len(images), self.x_dim, self.y_dim, self.c_dim))
    
         counter1 = 0

         for epoch in range(old_epoch, num_epoch):
 
             batch_idxs = len(images_lr) // self.batch_size
             for idx in range(0, batch_idxs):
# =============================================================================
#                  batch_images = images_lr[idx*self.batch_size:(idx+1)*self.batch_size]
#                  batch_labels = images_hr[idx*self.batch_size:(idx+1)*self.batch_size]
# =============================================================================
#                 batch_lr, batch_cb, batch_cr = rgb2ycbcr(images_lr[idx*self.batch_size:(idx+1)*self.batch_size])
#                 batch_hr, _, _ = rgb2ycbcr(images_hr[idx*self.batch_size:(idx+1)*self.batch_size])
                 
                 batch_lr = images_lr[idx*self.batch_size:(idx+1)*self.batch_size]
                 batch_hr = images_hr[idx*self.batch_size:(idx+1)*self.batch_size]
                 #batch_labels = labels[idx*self.batch_size:(idx+1)*self.batch_size]
                 
                 if self.is_ycbcr is True:
                     batch_lr = batch_lr[..., np.newaxis]
                     batch_hr = batch_hr[..., np.newaxis]
                     
                 self.sess.run(self.g_opt,feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 
                 self.sess.run(self.d_opt,feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 
                 #self.sess.run(self.opt, feed_dict = {self.x_lr: batch_images, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec}) 
               
                  # Display logs per display step
                 if (counter1+1) % display_step == 0:
                     g_loss, d_loss = self.sess.run((self.g_loss, self.d_loss), feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 
                     #loss = self.sess.run(self.loss, feed_dict = {self.x_lr: batch_images, self.x_hr: batch_labels, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec}) 
                   
                     print( "Epoch", '%d' % (epoch) ,\
                         "Step:" , '%d' %(counter1+1), \
                         "Sample:", '%d' % ((idx+1)*self.batch_size), \
                         "G Loss=", "{:.4f}".format(g_loss),\
                         "D Loss=", "{:.4f}".format(d_loss))
                         
                 if  (counter1) % 1000 == 0:
# =============================================================================
#                     if self.is_ycbcr is True:
#                         g_sample = self.generate_sr(rgb_images[idx*self.batch_size:(idx+1)*self.batch_size], self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)    
#                     else:
#                         g_sample = self.generate_sr(batch_images, self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)
# =============================================================================
                    batch_sr = self.generate_sr(batch_lr, self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)
                    if self.is_ycbcr is True:
                        batch_sr = ycbcr2rgb(batch_sr, cb[idx*self.batch_size:(idx+1)*self.batch_size], cr[idx*self.batch_size:(idx+1)*self.batch_size])
                    plot(batch_sr[0:16], name = 'Step'+str(counter1+1))
                     
                 counter1 += 1
             # save model
             if epoch != num_epoch and epoch % checkpoint_step == 0:                 
                 self.save_model(checkpoint_file, epoch)
                 print("model saved to {}".format(checkpoint_file))
 
         self.save_model(checkpoint_file, epoch + 1)   
         
    def save_model(self, checkpoint_path, epoch):
         """ saves the model to a file """
         self.saver.save(self.sess, checkpoint_path, global_step = epoch)
    
 
    def load_model(self, checkpoint_path):
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            #var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_path, ckpt_name))
        print("loading model: ",os.path.join(checkpoint_path, ckpt_name))
        
    def test (self, testset = 'Set14', test_dir = './data/SR_testset'):
        self.sess = tf.InteractiveSession()   
        self.saver = tf.train.Saver()
        
        #var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        #weight_initiallizer = tf.train.Saver(var_list)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        dirname = self.checkpoint_path
        ckpt = tf.train.get_checkpoint_state(dirname)
         
        if ckpt:
            self.load_model(dirname)
            if self.dataset_name =='mnist':
                images, labels = read_dataset()
                if images.shape[1] != self.x_dim:
                    images = tf.image.resize_images(images, (self.x_dim, self.y_dim)).eval()
                seed = np.random.randint(0,len(images))
                np.random.seed(seed)
                np.random.shuffle(images) 
                new_images = self.generate_sr(images[:16], self.x_dim, self.y_dim)
            else:              
                test_lr = read_data(test_dir, dataset = testset)
                c_dim = 1 if self.is_rgb is False else 3      
                if self.is_ycbcr is True:
                    test_hr, test_lr, test_cb, test_cr = prepare_data(test_lr, size = self.x_dim, sr_ratio = self.sr_ratio, is_rgb = self.is_rgb, is_shuffle = False, is_ycbcr = self.is_ycbcr, is_data_aug = False)
                    new_images_y = np.zeros((len(test_lr), self.x_dim *self.sr_ratio, self.y_dim * self.sr_ratio))
                new_images = np.zeros((len(test_lr), self.x_dim *self.sr_ratio, self.y_dim * self.sr_ratio, c_dim))
                if len(test_lr) < self.batch_size:
                    self.batch_size =  len(test_lr)
                for idx in range(0, len(test_lr)//self.batch_size):
                    test_sr = self.generate_sr(test_lr[idx*self.batch_size:(idx+1)*self.batch_size, :,:, np.newaxis], self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)
                    new_images_y[idx*self.batch_size:(idx+1)*self.batch_size,:,:] = test_sr
                    if self.is_ycbcr is True:
                        test_sr = ycbcr2rgb(test_sr, test_cb, test_cr)
                    new_images[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:] = test_sr
                    
            return new_images, new_images_y
            
    def encode(self,X):
        return self.sess.run(self.z_mu, feed_dict = {self.x_data: np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))})

    def reconstruct (self, X):
        return self.sess.run(self.x_new, feed_dict = {self.x_data: np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2])), self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})
        
    def close(self):
        self.sess.close()
        
    def coordinates(self, x_dim = 32, y_dim = 32, scale = 1.0):
        n_pixel = x_dim * y_dim
        x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
        y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
# =============================================================================
# 
#         if self.c_dim == 3:
#             x_mat = np.swapaxes(np.array([x_mat, x_mat, x_mat]), 0,2)
#             y_mat = np.swapaxes(np.array([y_mat, y_mat, y_mat]), 0,2)
#             r_mat = np.swapaxes(np.array([r_mat, r_mat, r_mat]), 0,2)
#             x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, self.c_dim)
#             y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, self.c_dim)
#             r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, self.c_dim)
# =============================================================================
        x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
        y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
        r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
        return x_mat, y_mat, r_mat
