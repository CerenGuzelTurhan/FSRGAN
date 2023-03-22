# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:50:37 2018

@author: Ceren
"""

import tensorflow as tf
import numpy as np
#from SR_CPPN import CPPN
#from srgan_cppn import SRGAN_CPPN
from fsrcnn import FSRCNN
from fsrgan import FSRGAN
from read_show_ops import *
from fsrresnet import FSRResNet
from fsrresgan import FSRResGAN
from deepsrresnet import DeepSRResNet
from deepsrresgan import DeepSRResGAN
import os
from ops2 import *
from read_show_ops import *

Flags = tf.app.flags

Flags.DEFINE_boolean('is_training', False, 'Training => True, Testing => False')
Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
Flags.DEFINE_string('model', 'FSRResNet', 'The task: FSRResGAN, FSRResnet')
# The data preparing operation
Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
Flags.DEFINE_string('data_dir', None, 'The directory of the input data')
Flags.DEFINE_boolean('is_data_aug', False, 'Many crops from single images')
Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
Flags.DEFINE_boolean('random_crop', False, 'Whether perform the random crop')
Flags.DEFINE_integer('org_size', 32, 'The crop size of the training image')
Flags.DEFINE_integer('num_resblock', 3, 'How many residual blocks are there in the generator')
# The content loss parameter
Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
# The training parameters
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 10000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', 1000, 'The max epoch for the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')
Flags.DEFINE_integer('sr_ratio', 2, 'The scaling factor')
Flags.DEFINE_boolean('is_rgb', True, 'Input Data is RGB or Gray level')
Flags.DEFINE_boolean('is_ycbcr', True, ' Input is converted to Y channel for processing or not')

Flags.DEFINE_boolean('pretrained_model', False, 'Pretrained model for initializing generator')

Flags.DEFINE_string('trainset', 'DIV2K', 'Training set name')
Flags.DEFINE_string('testset', 'BSDS100', 'Testing set name')
Flags.DEFINE_string('test_dir',  './data/SR_testset', 'testing sets directory')
Flags.DEFINE_boolean("valset", None, "Does validation set exist")


FLAGS = Flags.FLAGS

if FLAGS.trainset =='sr':
    FLAGS.data_dir = './data/SR_trainset'
    FLAGS.is_data_aug = True
elif FLAGS.trainset == 'mscoco':
    FLAGS.is_data_aug = False
    FLAGS.data_dir = 'D:/'
elif FLAGS.trainset == 'DIV2K':
    FLAGS.is_data_aug = False
    FLAGS.data_dir = './data/DIV2K_train'
    if FLAGS.valset:
        FLAGS.val_dir = './data/DIV2K_val'

FLAGS.checkpoint_path = './save/' + FLAGS.model +'/'+ FLAGS.trainset + '/x'+str(FLAGS.sr_ratio)
if FLAGS.pretrained_model is True:
    FLAGS.pretrained_model_path = './save/DeepSRResNet/' + FLAGS.trainset + '/x'+str(FLAGS.sr_ratio)
else:
    FLAGS.pretrained_model_path = None
if not os.path.exists(FLAGS.checkpoint_path):
    os.makedirs(FLAGS.checkpoint_path)

if FLAGS.model == 'FSRResNet':
    net = FSRResNet(FLAGS)
elif FLAGS.model == 'FSRResGAN':
    net = FSRResGAN(FLAGS)    
elif FLAGS.model == 'DeepSRResNet':
    net = DeepSRResNet(FLAGS)
elif FLAGS.model == 'DeepSRResGAN':
    net = DeepSRResGAN(FLAGS)  
        
if FLAGS.is_training is True:
    net.train()
    
else:
    if FLAGS.trainset == 'DIV2K':
        if FLAGS.is_ycbcr is True:            
            imgs, test_hr_rgb, test_lr_rgb = net.test() 
    else:
        imgs, imgs_y = net.test()
        test_imgs = read_data(FLAGS.test_dir, dataset = FLAGS.testset)
        test_hr_rgb, test_lr_rgb = prepare_data(test_imgs, size = FLAGS.org_size, sr_ratio = FLAGS.sr_ratio, is_rgb = FLAGS.is_rgb, is_shuffle = False, is_ycbcr = False, is_data_aug = FLAGS.is_data_aug)  
        if FLAGS.is_ycbcr is True:
            test_hr, test_lr, test_cb, test_cr = prepare_data(test_imgs, size = FLAGS.org_size, sr_ratio = FLAGS.sr_ratio, is_rgb = FLAGS.is_rgb, is_shuffle = False, is_ycbcr = FLAGS.is_ycbcr, is_data_aug = FLAGS.is_data_aug)
    
    show_id = int(np.sqrt(min(16, len(imgs))))**2
    seed = np.random.randint(0,len(imgs)-1)
    np.random.seed(seed)
    np.random.shuffle(imgs)
    np.random.seed(seed)
    np.random.shuffle(test_hr_rgb)

    print("SR images in %dx%d using: " %(FLAGS.org_size*FLAGS.sr_ratio, FLAGS.org_size*FLAGS.sr_ratio) , FLAGS.model)
    plot(imgs[:show_id]) 
    
    show_id = int(np.sqrt(min(16, len(test_hr_rgb))))**2
    grd_imgs = test_hr_rgb[0:show_id]
    org_imgs = test_lr_rgb[0:show_id]
    print("Original images:")
    plot(grd_imgs)

# =============================================================================
#     imgs_bicubic = tf.image.resize_images(test_lr_rgb,(FLAGS.org_size*FLAGS.sr_ratio, FLAGS.org_size*FLAGS.sr_ratio), tf.image.ResizeMethod.BICUBIC).eval()
#     print("Bicubic SR images in %dx%d:" %(FLAGS.org_size*FLAGS.sr_ratio, FLAGS.org_size*FLAGS.sr_ratio))
#     imgs_bicubic[imgs_bicubic < 0] = 0
#     plot(imgs_bicubic[:show_id])
# 
#     imgs_nearest =  tf.image.resize_images(test_lr_rgb,(FLAGS.org_size*FLAGS.sr_ratio, FLAGS.org_size*FLAGS.sr_ratio), tf.image.ResizeMethod.NEAREST_NEIGHBOR).eval()
#     print("Nearest SR images in %dx%d:" %(FLAGS.org_size*FLAGS.sr_ratio, FLAGS.org_size*FLAGS.sr_ratio))
#     plot(imgs_nearest[:show_id])
# =============================================================================
    if FLAGS.trainset == 'DIV2K':
        psnr_model = 0
        for i in range (0,len(imgs)//16):
            psnr_model += psnr(test_hr[i*16:(i+1)*16], imgs_y[i*16:(i+1)*16])
        psnr_model = psnr_model / len(range(len(imgs)//16))
        print(FLAGS.model +"  %.4f" %(psnr_model))
    else:
        psnr_model = psnr(test_hr, imgs_y)
        psnr_bicubic= psnr(test_hr_rgb, imgs_bicubic)
        psnr_nearest = psnr(test_hr_rgb, imgs_nearest)
        print("Methods  PSNR")
        print("---------------")
        print("Bicubic  %.4f" %(psnr_bicubic))
        print("Nearest  %.4f" %(psnr_nearest))
        print(FLAGS.model +"  %.4f" %(psnr_model))

   