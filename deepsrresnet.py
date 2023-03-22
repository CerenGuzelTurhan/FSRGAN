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
from ops2 import prelu, compute_psnr, fully_connected, batch_normal
from read_show_ops import read_data, read_dataset, loadImagenet, prepare_data, ycbcr2rgb, rgb2ycbcr, plot, prepare_cropped_data
from mnist_data import *
from bicubic_interp import bicubic_interp_2d


class DeepSRResNet():
    def __init__(self, FLAGS):
        
        self.x_dim = FLAGS.org_size
        self.y_dim = FLAGS.org_size
        self.dim = FLAGS.org_size
        
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate              
        self.checkpoint_path = FLAGS.checkpoint_path
        self.data_dir = FLAGS.data_dir
        if FLAGS.valset:
            self.val_dir = FLAGS.val_dir
        self.sr_ratio = FLAGS.sr_ratio
        self.keep_prob = 0.95
        self.momentum = 0.9
        self.beta1 = 0.9

        self.c_dim = 1 if not FLAGS.is_rgb else 3 if not FLAGS.is_ycbcr else 1
        self.is_ycbcr = FLAGS.is_ycbcr
        self.is_rgb = FLAGS.is_rgb
        self.is_data_aug = FLAGS.is_data_aug
        self.trainset = FLAGS.trainset
        self.testset = FLAGS.testset
        self.max_epoch = FLAGS.max_epoch
        self.m, self.s, self.d = FLAGS.num_resblock, 12, 32
        #lambda1, lambda2, lambda3 = 1e-5, 1, 1

        self.get_weights()
               
        self.x_lr = tf.placeholder(tf.float32, shape = [self.batch_size, self.x_dim,self.y_dim, self.c_dim])
        self.x_hr = tf.placeholder(tf.float32, shape = [self.batch_size, self.x_dim*self.sr_ratio,self.y_dim*self.sr_ratio, self.c_dim])            
        self.img_bicubic = tf.image.resize_bicubic(self.x_lr,(self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio))
        self.x_tilde = self.generator(self.x_lr,self.x_dim * self.sr_ratio, self.y_dim * self.sr_ratio)
# =============================================================================

        
# =============================================================================
      
        #self.d_x_tilde, self.f_x_tilde, self.d_tilde_logit = self.discriminator(self.x_tilde)
        #self.d_x, self.f_x, self.d_logit = self.discriminator(self.x_hr, reuse = True)

        #self.ll_loss = tf.reduce_mean(tf.square(self.f_x-self.f_x_tilde))

        #self.d_loss_real = tf.reduce_mean(tf.nn.l2_loss(self.d_logit- tf.ones_like(self.d_logit)))
        #self.d_loss_tilde = tf.reduce_mean(tf.nn.l2_loss(self.d_tilde_logit - tf.zeros_like(self.d_tilde_logit)))
        #self.d_loss = self.d_loss_real + self.d_loss_tilde
        
        #self.g_loss_tilde = binary_cross_entropy_with_logits(tf.ones_like(self.d_x_tilde), self.d_x_tilde)
        #g_loss = tf.reduce_mean(tf.nn.l2_loss(self.d_tilde_logit - tf.ones_like(self.d_tilde_logit)))
        #self.pixel_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_tilde - self.x_hr), reduction_indices=0), name = 'pixel_loss')
        #self.g_loss = lambda1 * g_loss + lambda2* self.ll_loss  + lambda3* self.pixel_loss

        #self.g_loss = self.g_loss_tilde + self.ll_loss
        #self.gan_loss = self.g_loss + self.d_loss
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_tilde - self.x_hr), reduction_indices=0), name = 'pixel_loss')
        self.psnr = compute_psnr(self.x_hr, self.x_tilde)
        #self.loss = 0.6*self.pixel_loss + 0.4* self.gan_loss
        
        #self.log_vars.append(("Generator Loss", self.g_loss))
        #self.log_vars.append(("Discriminator Loss", self.d_loss))
        
        t_vars = tf.trainable_variables()
        #self.d_vars = [var for var in t_vars if 'd_' in var.name]
        #self.g_vars = [var for var in t_vars if 'g_' in var.name]
        #global_step = tf.contrib.framework.get_or_create_global_step()
        #self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate, staircase=False)
        self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1= self.beta1) \
                          .minimize(self.loss, var_list=t_vars)
                          
        tf.summary.scalar('MSE Loss', self.loss)
        tf.summary.scalar('PSNR', self.psnr)
        #tf.summary.scalar('learning_rate', self.learning_rate)

        #self.opt =  tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss)
# =============================================================================
#         self.d_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
#               .minimize(self.d_loss, var_list=self.d_vars)
#         self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
#               .minimize(self.g_loss, var_list=self.g_vars)
# =============================================================================
              
        #var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        
#        for k,v in self.log_vars:
#            tf.summary.scalar(k, v)
#    
    def get_weights(self):
        self.weights = {
          'conv_w1': tf.Variable(tf.random_normal([5, 5, self.c_dim , self.s ], stddev=0.0378, dtype=tf.float32), name='g_w1'),
          'conv_w2': tf.Variable(tf.random_normal([1, 1, self.s , self.d ], stddev=0.3536, dtype=tf.float32), name='g_w2'),
          'conv_w3': tf.Variable(tf.random_normal([3, 3, self.d , self.d ], stddev=0.1179, dtype=tf.float32), name='g_w3'),
          'conv_w4': tf.Variable(tf.random_normal([3, 3, self.d , self.d ], stddev=0.1179, dtype=tf.float32), name='g_w4'),
          'conv_w5': tf.Variable(tf.random_normal([3, 3, self.c_dim , self.c_dim ], stddev=0.1179, dtype=tf.float32), name='g_w5'),
#          'conv_w6': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.3536, dtype=tf.float32), name='g_w6'),
#          'conv_w7': tf.Variable(tf.random_normal([3, 3, 64, self.c_dim], stddev=0.3536, dtype=tf.float32), name='g_w7'),
          'deconv_w1': tf.Variable(tf.random_normal([4, 4, self.d , self.d ], stddev=0.0001, dtype=tf.float32), name='g_dw1'),
          'deconv_w2': tf.Variable(tf.random_normal([4, 4, self.d , self.d ], stddev=0.0001, dtype=tf.float32), name='g_dw2'),
          'deconv_w3': tf.Variable(tf.random_normal([4, 4, self.c_dim, self.d ], stddev=0.0001, dtype=tf.float32), name='g_dw3')
        }

        #expand_bias, deconv_bias = 'b{}'.format(self.m + 3), 'b{}'.format(self.m + 4)
        self.biases = {
          'conv_b1': tf.Variable(tf.zeros([self.s ]), name='g_b1'),
          'conv_b2': tf.Variable(tf.zeros([self.d ]), name='g_b2'),
          'conv_b3': tf.Variable(tf.zeros([self.d ]), name='g_b3'),
          'conv_b4': tf.Variable(tf.zeros([self.d ]), name='g_b4'),
          'conv_b5': tf.Variable(tf.zeros([self.c_dim ]), name='g_b5'),
#          'conv_b6': tf.Variable(tf.zeros([64]), name='g_b6'),
#          'conv_b7': tf.Variable(tf.zeros([self.c_dim]), name='g_b7'),
          'deconv_b1': tf.Variable(tf.zeros([self.d ]), name='g_db1'),
          'deconv_b2': tf.Variable(tf.zeros([self.d ]), name='g_db2'),
          'deconv_b3': tf.Variable(tf.zeros([self.c_dim]), name='g_db3')
        }

        # Create the m mapping layers weights/biases
        for i in range(3, self.m + 3):
          weight_name, bias_name = 'conv_w3_{}'.format(i), 'conv_b3_{}'.format(i)
          weight_name2, bias_name2 = 'conv_w3_{}'.format(i+1), 'conv_b3_{}'.format(i+1)
          self.weights[weight_name] = tf.Variable(tf.random_normal([3, 3, self.d , self.d ], stddev=0.1179, dtype=tf.float32), name='g_w3_{}'.format(i))
          self.biases[bias_name] = tf.Variable(tf.zeros([self.d ]), name='g_b3_{}'.format(i))  
          self.weights[weight_name2] = tf.Variable(tf.random_normal([3, 3, self.d , self.d ], stddev=0.1179, dtype=tf.float32), name='g_w3_{}'.format(i+1))
          self.biases[bias_name2] = tf.Variable(tf.zeros([self.d ]), name='g_b3_{}'.format(i+1))   
    
    def upsample(self, x, factor=2):
            _, h, w, _ = x.get_shape().as_list()
            x = tf.image.resize_nearest_neighbor(x, [factor * h, factor * w])
            return x      
        
    def generator(self, img, x_dim = 64, y_dim = 64, reuse = False): # OR DECODER
        with tf.variable_scope('generator') as scope:
            if reuse == True:
                scope.reuse_variables() 
            img = tf.cast(img, tf.float32)   
            self.conv1 = prelu(tf.nn.conv2d(img, self.weights['conv_w1'], strides=[1,1,1,1], padding='SAME') + self.biases['conv_b1'], 1) 
            self.conv2 = prelu(tf.nn.conv2d(self.conv1, self.weights['conv_w2'], strides=[1,1,1,1], padding='SAME') + self.biases['conv_b2'], 2)
            # Mapping (# mapping layers = m)
            #ResBlock
            self.prev_layer, m = self.conv2, self.m
            for i in range(3, m + 3):
                self.resblock_input = self.prev_layer
                weights1, weights2, biases1, biases2 = self.weights['conv_w3_{}'.format(i)], self.weights['conv_w3_{}'.format(i+1)], self.biases['conv_b3_{}'.format(i)], self.biases['conv_b3_{}'.format(i+1)]
                self.prev_layer_1 = prelu(tf.nn.conv2d(self.prev_layer, weights1, strides=[1,1,1,1], padding='SAME') + biases1, i)
                self.prev_layer_2 = tf.nn.conv2d(self.prev_layer_1, weights2, strides=[1,1,1,1], padding='SAME') + biases2
                self.prev_layer = self.prev_layer_2 + self.resblock_input 
            
            self.deconv1 = prelu(tf.nn.conv2d_transpose(self.prev_layer , self.weights['deconv_w1'], [self.batch_size, self.x_dim*2, self.y_dim*2, self.d ], strides=[1,2,2,1], padding='SAME') + self.biases['deconv_b1'], m + 4)
            #self.bicubic_x2 = bicubic_interp_2d(self.prev_layer, [self.x_dim*2, self.y_dim*2], endpoint = True) #tf.image.resize_images(self.prev_layer,(self.x_dim*2, self.y_dim*2), tf.image.ResizeMethod.BICUBIC)
            self.conv3 = prelu(tf.nn.conv2d(self.deconv1, self.weights['conv_w3'], strides=[1,1,1,1], padding='SAME') + self.biases['conv_b3'], m + 5)
            self.bicubic_x2 = self.upsample(self.prev_layer)
            self.conv3 = self.conv3  +  self.bicubic_x2           
            
            
            self.deconv2 = prelu(tf.nn.conv2d_transpose(self.conv3, self.weights['deconv_w2'], [self.batch_size, self.x_dim*4, self.y_dim*4, self.d ], strides=[1,2,2,1], padding='SAME') + self.biases['deconv_b2'], (m + 6))
            self.conv4 = prelu(tf.nn.conv2d(self.deconv2, self.weights['conv_w4'], strides=[1,1,1,1], padding='SAME') + self.biases['conv_b4'], m + 7)
            #self.bicubic_x4 = bicubic_interp_2d(self.conv3, [self.x_dim*4, self.y_dim*4], endpoint = True) #tf.image.resize_images(self.conv3,(self.x_dim*4, self.y_dim*4), tf.image.ResizeMethod.BICUBIC)
            self.bicubic_x4 = self.upsample(self.conv3)
            self.conv4  = self.conv4   +  self.bicubic_x4
                    
            self.deconv3 = tf.nn.conv2d_transpose(self.conv4, self.weights['deconv_w3'], [self.batch_size, self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio, self.c_dim], strides=[1,2,2,1], padding='SAME') + self.biases['deconv_b3']
            self.conv5 = prelu(tf.nn.conv2d(self.deconv3, self.weights['conv_w5'], strides=[1,1,1,1], padding='SAME') + self.biases['conv_b5'], m + 8)
            self.bicubic_x8 = self.img_bicubic #bicubic_interp_2d(img, [self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio], endpoint = True) #tf.image.resize_images(img,(self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio), tf.image.ResizeMethod.BICUBIC)
            self.conv5 = self.conv5  + self.bicubic_x8
            
           # self.conv5 = ((self.conv5 - tf.reduce_min(self.conv5)) * 1) / (tf.reduce_max(self.conv5) - tf.reduce_min(self.conv5))
                             
            return self.conv5                                                        
                                                               
        
#    def discriminator(self, img, reuse = False):
#        with tf.variable_scope('discriminator') as scope:
#            if reuse == True:
#                scope.reuse_variables()  
#              
#            # input part        
#            self.h1 = lrelu(conv2d (img, output_dim = 64, k_h = 4, k_w = 4, name='d_h1_conv1'))  
#            self.h2 = lrelu(batch_normal(conv2d (self.h1, output_dim = 64, k_h = 4, k_w = 4, name='d_h2_conv2'), scope = 'd_bn1')) 
#            self.h3 = lrelu(batch_normal(conv2d (self.h2, output_dim = 64,  k_h = 4, k_w = 4, name='d_h3_conv3'), scope = 'd_bn2')) 
#            self.h4 = lrelu(batch_normal(conv2d (self.h3, output_dim = 64,  k_h = 4, k_w = 4,name='d_h4_conv4'), scope='d_bn3')) # 4x4x128
#            self.h4 = tf.reshape(self.h4, [self.batch_size, -1])
#            self.h5_logit = linear(self.h4, output_size = 1, scope = 'd_h5_lin1')
#            self.result = tf.nn.sigmoid(self.h5_logit)
#            return self.result, self.h4, self.h5_logit
           
    def generate_sr(self, img, x_dim = 64, y_dim = 64, scale = 5.0):

        model = self.generator(img, x_dim, y_dim, reuse = True)    
        images = self.sess.run(model, feed_dict = {self.x_lr: img})
        if self.c_dim != 3:
            images = images.reshape(images.shape[:-1])
        return images
          
    def train(self, display_step = 1, checkpoint_step = 10):
         init = tf.initialize_all_variables()        
         self.sess = tf.InteractiveSession()        
         self.sess.run(init)  
         self.saver = tf.train.Saver()
         ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
         checkpoint_file = os.path.join(self.checkpoint_path , 'model.ckpt')
         old_epoch, epoch = 0, 0
         if ckpt:
             self.load_model(self.checkpoint_path)
             print ("Model is loaded!")
             ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
             ckpt_epoch = int(ckpt_name.split('-')[1])
             old_epoch = ckpt_epoch
         if self.trainset == 'mnist':
             images_hr, labels = read_dataset()
             if images_hr.shape[1] != self.x_dim*self.sr_ratio:
                 images_hr = tf.image.resize_images(images_hr, (self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)).eval()
             images_lr = tf.image.resize_images(images_hr, (self.x_dim, self.y_dim)).eval()    
         elif self.trainset == 'imagenet':
             images_hr, images_lr, cb, cr =  loadImagenet(self.data_dir, self.x_dim *self.sr_ratio, self.sr_ratio, 2 , self.is_ycbcr)
         elif self.trainset == 'DIV2K':
             if self.is_ycbcr is True:
                 images_hr, images_lr, cb, cr, _, _  = prepare_cropped_data(lr_size = self.x_dim)
             else:
                images_hr, images_lr = prepare_cropped_data(lr_size = self.x_dim, is_ycbcr = self.is_ycbcr)
         else:
             imgs = read_data(self.data_dir, self.trainset)
             images_hr, images_lr, cb, cr  = prepare_data(imgs, size = self.x_dim, sr_ratio = self.sr_ratio, is_rgb = self.is_rgb, is_shuffle = False,  is_data_aug = self.is_data_aug) 

    
         counter1 = 0

         for epoch in range(old_epoch, self.max_epoch):
 
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
                     
                 #self.sess.run(self.g_opt,feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 
                 #self.sess.run(self.d_opt,feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 
                 self.sess.run(self.opt, feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 
               
                  # Display logs per display step
                 if (counter1+1) % display_step == 0:
                     #g_loss, d_loss = self.sess.run((self.g_loss, self.d_loss), feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 
                     loss, psnr = self.sess.run([self.loss, self.psnr], feed_dict = {self.x_lr: batch_lr, self.x_hr: batch_hr}) 

                     print( "Epoch", '%d' % (epoch),  "Step:" , '%d' %(counter1+1),  "Sample:", '%d' % ((idx+1)*self.batch_size), "Loss=", "{:.4f}".format(loss), "PSNR:" ,'%.2f' %psnr)
#                     print( "Epoch", '%d' % (epoch) ,\
#                         "Step:" , '%d' %(counter1+1), \
#                         "Sample:", '%d' % ((idx+1)*self.batch_size), \
#                         "G Loss=", "{:.4f}".format(g_loss),\
#                         "D Loss=", "{:.4f}".format(d_loss))
                         
                 if  (counter1) % 1000 == 0:
# =============================================================================
#                     if self.is_ycbcr is True:
#                         g_sample = self.generate_sr(rgb_images[idx*self.batch_size:(idx+1)*self.batch_size], self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)    
#                     else:
#                         g_sample = self.generate_sr(batch_images, self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)
# =============================================================================
                    batch_sr = self.generate_sr(batch_lr, self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)
                    batch_sr = ((batch_sr - np.min(batch_sr)) * 1) / (np.max(batch_sr) - np.min(batch_sr))
                    if self.is_ycbcr is True:
                        batch_sr = ycbcr2rgb(batch_sr, cb[idx*self.batch_size:(idx+1)*self.batch_size], cr[idx*self.batch_size:(idx+1)*self.batch_size])                   
                       
# =============================================================================
                    
                    show_id = int(np.sqrt(min(16, self.batch_size)))**2
                    plot(batch_sr[0:show_id], name = 'Step'+str(counter1+1))
                     
                 counter1 += 1
             # save model
             if epoch != self.max_epoch and epoch % checkpoint_step == 0:                 
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
        
    def test (self, test_dir = './data/SR_testset'):

        self.sess = tf.InteractiveSession()   
        self.saver = tf.train.Saver()
        
        #var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        #weight_initiallizer = tf.train.Saver(var_list)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        dirname = self.checkpoint_path
#        ckpt = tf.train.get_checkpoint_state(dirname)
#        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
#        weight_initiallizer.restore(self.sess, os.path.join(dirname, ckpt_name))
        self.load_model(dirname) 
        if self.trainset =='mnist':
            images, labels = read_dataset()
            if images.shape[1] != self.x_dim:
                images = tf.image.resize_images(images, (self.x_dim, self.y_dim)).eval()
            seed = np.random.randint(0,len(images))
            np.random.seed(seed)
            np.random.shuffle(images) 
            new_images = self.generate_sr(images[:16], self.x_dim, self.y_dim)
        elif self.trainset == 'DIV2K':
            if self.is_ycbcr is True:
                test_hr, test_lr, test_cb, test_cr, test_hr_rgb, test_lr_rgb = prepare_cropped_data(lr_size = self.x_dim)
                new_images_y = np.zeros((len(test_lr), self.x_dim *self.sr_ratio, self.y_dim * self.sr_ratio))
                
            else:
                test_hr, test_lr = prepare_cropped_data(lr_size = self.x_dim, is_ycbcr = self.is_ycbcr) 
            new_images = np.zeros((len(test_lr), self.x_dim *self.sr_ratio, self.y_dim * self.sr_ratio, 3), dtype = np.uint8)
            
            if len(test_lr) < self.batch_size:
                self.batch_size =  len(test_lr)
            for idx in range(0, len(test_lr)//self.batch_size):
                test_sr = self.generate_sr(test_lr[idx*self.batch_size:(idx+1)*self.batch_size, :,:, np.newaxis], self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)
                new_images[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:] = test_sr
                if self.is_ycbcr is True:
                    test_sr = ycbcr2rgb(test_sr, test_cb, test_cr)
                    new_images_y[idx*self.batch_size:(idx+1)*self.batch_size,:,:] = test_sr              
                    return np.array(new_images, dtype = np.uint8), new_images_y, test_hr, test_lr, test_cb, test_cr, test_hr_rgb, test_lr_rgb              
                else:
                     return np.array(new_images*255, dtype = np.uint8), test_hr, test_lr              
        else:              
            test_lr = read_data(test_dir, dataset = self.testset)
            c_dim = 1 if self.is_rgb is False else 3      
            if self.is_ycbcr is True:
                test_hr, test_lr, test_cb, test_cr = prepare_data(test_lr, size = self.x_dim, sr_ratio = self.sr_ratio, is_rgb = self.is_rgb, is_shuffle = False, is_ycbcr = self.is_ycbcr, is_data_aug = False)
                new_images_y = np.zeros((len(test_lr), self.x_dim *self.sr_ratio, self.y_dim * self.sr_ratio))
            new_images = np.zeros((len(test_lr), self.x_dim *self.sr_ratio, self.y_dim * self.sr_ratio, c_dim), dtype = np.uint8)
            if len(test_lr) < self.batch_size:
                self.batch_size =  len(test_lr)
            for idx in range(0, len(test_lr)//self.batch_size):
                test_sr = self.generate_sr(test_lr[idx*self.batch_size:(idx+1)*self.batch_size, :,:, np.newaxis], self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)
                new_images_y[idx*self.batch_size:(idx+1)*self.batch_size,:,:] = test_sr
                if self.is_ycbcr is True:
                    test_sr = ycbcr2rgb(test_sr, test_cb, test_cr)
                new_images[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:] = test_sr
                
            return np.array(new_images, dtype = np.uint8), new_images_y
            
    def close(self):
        self.sess.close()

