# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:36:51 2018

@author: Ceren
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from ops2 import *
from read_show_ops import *
from mnist_data import *

class FSRCNN(object):
    def __init__(self,x_dim = 64, y_dim = 64,  batch_size = 128,  is_rgb = True, learning_rate = 0.001, momentum = 0.9, checkpoint_path = './save/fsrcnn',  is_data_aug = True, data_dir = None, dataset_name = 'sr', sr_ratio = 1, m = 4, s = 12, d = 56, is_ycbcr = False, dataset_index = None):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.dim = x_dim
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dataset_name = dataset_name
        self.dataset_index = dataset_index
              
        self.checkpoint_path = checkpoint_path
        
        self.log_vars = []
        self.c_dim = 1 if not is_rgb else 3 if not is_ycbcr else 1
        self.data_dir = data_dir
        self.sr_ratio = sr_ratio
        self.is_rgb = is_rgb
        self.is_ycbcr = is_ycbcr
        self.is_data_aug = is_data_aug
        
        self.x_lr = tf.placeholder(tf.float32, shape = [None, x_dim, y_dim, self.c_dim ])
        self.x_hr = tf.placeholder(tf.float32, shape = [None, x_dim*sr_ratio, y_dim*sr_ratio, self.c_dim ])
        self.m, self.s, self.d = m, s, d

        self.get_weights()
        self.x_new = self.model(self.x_lr, x_dim * sr_ratio, y_dim * sr_ratio) #reconstructed images
        
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_new - self.x_hr), reduction_indices=0), name = 'loss')
        #self.loss = tf.reduce_mean(tf.squared_difference(self.x_new, self.x_hr), name='loss')
# =============================================================================
#         self.reconstr_loss = -tf.reduce_sum(self.x_data * tf.log(1e-10 + self.x_new)
#                        + (1-self.x_data) * tf.log(1e-10 + 1 - self.x_new), 1)
#         
#         self.loss = tf.reduce_mean(self.reconstr_loss)/self.batch_size
# =============================================================================
        self.log_vars.append(("Loss", self.loss))
        self.opt =  tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss)
#        self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1= momentum) \
#                          .minimize(self.loss)
        init = tf.initialize_all_variables()
        
        self.sess = tf.InteractiveSession()
        
        self.sess.run(init)
        self.saver = tf.train.Saver(tf.all_variables())
        
        for k,v in self.log_vars:
            tf.summary.scalar(k, v)
        
    def get_weights(self):
        expand_weight, deconv_weight = 'w{}'.format(self.m + 3), 'w{}'.format(self.m + 4)
        self.weights = {
          'w1': tf.Variable(tf.random_normal([5, 5, self.c_dim , self.d], stddev=0.0378, dtype=tf.float32), name='w1'),
          'w2': tf.Variable(tf.random_normal([1, 1, self.d, self.s], stddev=0.3536, dtype=tf.float32), name='w2'),
          expand_weight: tf.Variable(tf.random_normal([1, 1, self.s, self.d], stddev=0.189, dtype=tf.float32), name=expand_weight),
          deconv_weight: tf.Variable(tf.random_normal([9, 9, self.c_dim, self.d], stddev=0.0001, dtype=tf.float32), name=deconv_weight)
        }

        expand_bias, deconv_bias = 'b{}'.format(self.m + 3), 'b{}'.format(self.m + 4)
        self.biases = {
          'b1': tf.Variable(tf.zeros([self.d]), name='b1'),
          'b2': tf.Variable(tf.zeros([self.s]), name='b2'),
          expand_bias: tf.Variable(tf.zeros([self.d]), name=expand_bias),
          deconv_bias: tf.Variable(tf.zeros([self.c_dim]), name=deconv_bias)
        }

        # Create the m mapping layers weights/biases
        for i in range(3, self.m + 3):
          weight_name, bias_name = 'w{}'.format(i), 'b{}'.format(i)
          self.weights[weight_name] = tf.Variable(tf.random_normal([3, 3, self.s, self.s], stddev=0.1179, dtype=tf.float32), name=weight_name)
          self.biases[bias_name] = tf.Variable(tf.zeros([self.s]), name=bias_name)

        
    def model(self, img, x_dim = 64, y_dim = 64, reuse = False): 

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
            #tf.nn.conv2d_backprop_input(deconv_output, deconv_weights, self.conv_expand, strides=deconv_stride, padding='VALID') + deconv_biases
            #◘

            return self.conv_deconv
        
        
        
                    
    def generate_sr(self, img, x_dim = 64, y_dim = 64):
# =============================================================================
#         is_ycbcr = False
#         if img.shape[3] == 3:
#             is_ycbcr = True
#             img, cb, cr = rgb2ycbcr(img)
#             img = img[..., np.newaxis]
# =============================================================================
        model = self.model(img, x_dim, y_dim, reuse = True)    
        images = self.sess.run(model)
        images = images.reshape(images.shape[:-1])
# =============================================================================
#         if is_ycbcr is True:
#             images = ycbcr2rgb(np.reshape(images, images.shape[:-1]), cb, cr)
# =============================================================================
        return images
         

    def train(self, num_epoch = 100, display_step = 1, checkpoint_step = 100):
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
             images_hr, images_lr, cb, cr = prepare_data(imgs, size = self.x_dim, sr_ratio = self.sr_ratio, is_rgb = self.is_rgb, is_shuffle = False, is_ycbcr = self.is_ycbcr, is_data_aug = self.is_data_aug)
         counter1 = 0
         
         for epoch in range(old_epoch, num_epoch):
 
             batch_idxs = len(images_hr) // self.batch_size
             for idx in range(0, batch_idxs):
                 batch_lr = images_lr[idx*self.batch_size:(idx+1)*self.batch_size]
                 batch_hr = images_hr[idx*self.batch_size:(idx+1)*self.batch_size]
                 #♦batch_labels = labels[idx*self.batch_size:(idx+1)*self.batch_size]
                 
                 if self.is_ycbcr is True:
                     batch_lr = batch_lr[..., np.newaxis]
                     batch_hr = batch_hr[..., np.newaxis]

                 self.sess.run(self.opt, feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 
               
                  # Display logs per display step
                 if (counter1+1) % display_step == 0:
                     loss = self.sess.run(self.loss, feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 

                     
                     print( "Epoch", '%d' % (epoch) ,\
                         "Step:" , '%d' %(counter1+1), \
                         "Sample:", '%d' % ((idx+1)*self.batch_size), \
                         "loss=", "{:.4f}".format(loss))

                 if  (counter1) % 1000 == 0:
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
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_path, ckpt_name))
        print("loading model: ",os.path.join(checkpoint_path, ckpt_name))
        
    def test (self, testset = 'Set14', test_dir = './data/SR_testset'):
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
                    test_hr, test_lr, test_cb, test_cr = prepare_data(test_lr, size = self.x_dim, sr_ratio = self.sr_ratio, is_rgb = self.is_rgb, is_shuffle = False, is_ycbcr = self.is_ycbcr, is_data_aug=False)
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

    def close(self):
        self.sess.close()
        

