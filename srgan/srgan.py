import tensorflow as tf
import numpy as np
import sys
import os
from srgan.layer import *
from srgan.vgg19 import VGG19
#import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from PIL import Image
from scipy.misc import imresize
        
class SRGAN:
    def __init__(self, batch_size, img_size, sr_ratio, checkpoint_path, data_dir):
        self.learning_rate = 1e-3
        #batch_size = 16
        self.vgg_model = './srgan/vgg/vgg19.npy'
        self.batch_size = batch_size
        self.sr_ratio = sr_ratio
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.x = tf.placeholder(tf.float32, [self.batch_size, img_size, img_size, 3]) #HR image
        self.is_training = tf.placeholder(tf.bool, [])
        
        self.vgg = VGG19(None, modelPath=self.vgg_model)
        self.downscaled = self.downscale(self.x)
        self.imitation = self.generator(self.downscaled, self.is_training, False)
        self.real_output = self.discriminator(self.x, self.is_training, False)
        self.fake_output = self.discriminator(self.imitation, self.is_training, True)
        self.g_loss, self.d_loss = self.inference_losses(
            self.x, self.imitation, self.real_output, self.fake_output)
       
        global_step = tf.Variable(0, name='global_step', trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.g_train_op = opt.minimize(
            self.g_loss, global_step=global_step, var_list=self.g_variables)
        self.d_train_op = opt.minimize(
            self.d_loss, global_step=global_step, var_list=self.d_variables)
        
        self.psnr = compute_psnr(self.x, self.imitation)

    
    def generator(self, x, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                x = deconv_layer(
                    x, [3, 3, 64, 3], [self.batch_size, 24, 24, 64], 1)
            x = tf.nn.relu(x)
            shortcut = x
            for i in range(5):
                mid = x
                with tf.variable_scope('block{}a'.format(i+1)):
                    x = deconv_layer(
                        x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('block{}b'.format(i+1)):
                    x = deconv_layer(
                        x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                    x = batch_normalize(x, is_training)
                x = tf.add(x, mid)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(
                    x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.add(x, shortcut)
            with tf.variable_scope('deconv3'):
                x = deconv_layer(
                    x, [3, 3, 256, 64], [self.batch_size, 24, 24, 256], 1)
                x = pixel_shuffle_layer(x, 2, 64) # n_split = 256 / 2 ** 2
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv4'):
                x = deconv_layer(
                    x, [3, 3, 64, 64], [self.batch_size, 48, 48, 64], 1)
                x = pixel_shuffle_layer(x, 2, 16)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv5'):
                x = deconv_layer(
                    x, [3, 3, 3, 16], [self.batch_size, 96, 96, 3], 1)

        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return x


    def discriminator(self, x, is_training, reuse):
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = lrelu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 64], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 128], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 512, 512], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            x = flatten_layer(x)
            with tf.variable_scope('fc'):
                x = full_connection_layer(x, 1024)
                x = lrelu(x)
            with tf.variable_scope('softmax'):
                x = full_connection_layer(x, 1)
                
        self.d_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return x

    def downscale(self, x):
        K = 4
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled

    def inference_losses(self, x, imitation, true_output, fake_output):
        def inference_content_loss(x, imitation):
            _, x_phi = self.vgg.buildCNN(
                x) # First
            tf.get_variable_scope().reuse_variables()
            _, imitation_phi = self.vgg.buildCNN(
                imitation) # Second

            content_loss = None
            for i in range(len(x_phi)):
                l2_loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                if content_loss is None:
                    content_loss = l2_loss
                else:
                    content_loss = content_loss + l2_loss
            return tf.reduce_mean(content_loss)

        def inference_adversarial_loss(real_output, fake_output):
            alpha = 1e-5
            g_loss = tf.reduce_mean(
                tf.nn.l2_loss(fake_output - tf.ones_like(fake_output)))
            d_loss_real = tf.reduce_mean(
                tf.nn.l2_loss(real_output - tf.ones_like(true_output)))
            d_loss_fake = tf.reduce_mean(
                tf.nn.l2_loss(fake_output + tf.zeros_like(fake_output)))
            d_loss = d_loss_real + d_loss_fake
            return (g_loss * alpha, d_loss * alpha)

        def inference_adversarial_loss_with_sigmoid(real_output, fake_output):
            alpha = 1e-3
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_output),
                logits=fake_output)
            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_output),
                logits=real_output)
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_output),
                logits=fake_output)
            d_loss = d_loss_real + d_loss_fake
            return (g_loss * alpha, d_loss * alpha)

        content_loss = inference_content_loss(x, imitation)
        generator_loss, discriminator_loss = (
            inference_adversarial_loss(true_output, fake_output))
        g_loss = content_loss + generator_loss
        d_loss = discriminator_loss
        return (g_loss, d_loss)
    
    def train(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer() 
        self.sess.run(init)

        # Restore the VGG-19 network
        var = tf.global_variables()
        self.vgg.loadModel(self.sess)
#        vgg_var = [var_ for var_ in var if "vgg19" in var_.name]
#        saver = tf.train.Saver(vgg_var)
#        saver.restore(self.sess, self.vgg_model)

        # Restore the SRGAN network
        if tf.train.get_checkpoint_state(self.checkpoint_path):
            checkpoint_file = os.path.join(self.checkpoint_path , 'model.ckpt')
            ckpt_name = os.path.basename(checkpoint_file.model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(self.sess, os.path.join(self.checkpoint_path, ckpt_name))
            
        # Load the data

        imgs = read_data(self.data_dir, self.dataset_name)
        images_hr, images_lr  = prepare_data(imgs, size = int(self.img_dim/self.sr_ratio), sr_ratio = self.sr_ratio, is_rgb = True, is_shuffle = False,  is_data_aug = False) 

        # Train the SRGAN model
        n_iter = int(len(images_hr) / self.batch_size)
        while True:
            epoch = int(self.sess.run(self.global_step) / n_iter / 2) + 1
            print('epoch:', epoch)
            np.random.shuffle(images_hr)
            for i in tqdm(range(n_iter)):
                x_batch = self.normalize(images_hr[i*self.batch_size:(i+1)*self.batch_size])
                self.sess.run(
                    [self.g_train_op, self.d_train_op],
                    feed_dict={self.x: x_batch, self.is_training: True})
                print( "Epoch", '%d' % (epoch),  "Iter:" , '%d' %(n_iter+1),  "Sample:", '%d' % ((i+1)*self.batch_size), " G Loss=", "{:.4f}".format(self.g_loss), " D Loss=", "{:.4f}".format(self.d_loss), "PSNR:" ,'%.2f' %self.psnr)
    
                if n_iter%10 == 0:# Validate
                    raw = self.normalize(images_hr[:self.batch_size])
                    mos, fake = self.sess.run(
                        [self.downscaled, self.imitation],
                        feed_dict={self.x: raw, self.is_training: False})
                    self.plot(fake)
                    #self.save_img([mos, fake, raw], ['Input', 'Output', 'Ground Truth'], epoch)
        
            # Save the model
            saver = tf.train.Saver()
            saver.save(self.sess, self.checkpoint_path, global_step = epoch)
            
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
                    
            test_lr = read_data(test_dir, dataset = testset)
            c_dim = 1 if self.is_rgb is False else 3                     
            new_images = np.zeros((len(test_lr), self.x_dim *self.sr_ratio, self.y_dim * self.sr_ratio, c_dim))
            if len(test_lr) < self.batch_size:
                self.batch_size =  len(test_lr)
            for idx in range(0, len(test_lr)//self.batch_size):
                test_sr = self.generate_sr(test_lr[idx*self.batch_size:(idx+1)*self.batch_size, :,:, np.newaxis], self.x_dim*self.sr_ratio, self.y_dim*self.sr_ratio)  
                new_images[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:] = test_sr
                    
            return new_images

# =============================================================================
#     def save_img(self,imgs, label, epoch):
#         for i in range(self.batch_size):
#             fig = plt.figure()
#             for j, img in enumerate(imgs):
#                 im = np.uint8((img[i]+1)*127.5)
#                 im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#                 fig.add_subplot(1, len(imgs), j+1)
#                 plt.imshow(im)
#                 plt.tick_params(labelbottom='off')
#                 plt.tick_params(labelleft='off')
#                 plt.gca().get_xaxis().set_ticks_position('none')
#                 plt.gca().get_yaxis().set_ticks_position('none')
#                 plt.xlabel(label[j])
#             seq_ = "{0:09d}".format(i+1)
#             epoch_ = "{0:09d}".format(epoch)
#             path = os.path.join('result', seq_, '{}.jpg'.format(epoch_))
#             if os.path.exists(os.path.join('result', seq_)) == False:
#                 os.mkdir(os.path.join('result', seq_))
#             plt.savefig(path)
#             plt.close()
# =============================================================================
            
    def plot(self, samples):

        is_rgb = True
        if len(samples.shape)>3 and samples.shape[3] == 3:
            nsample, x_dim, y_dim, c_dim = samples.shape
        else:
           nsample, x_dim, y_dim = samples.shape[:3]
           is_rgb = False
           
        fig_size = int(np.sqrt(nsample))
        fig = plt.figure(figsize=(fig_size, fig_size))
        gs = gridspec.GridSpec(fig_size, fig_size)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            if is_rgb is True:
                plt.imshow(sample.reshape(x_dim, y_dim,c_dim), cmap = 'gray')
            else:
                plt.imshow(sample.reshape(x_dim, y_dim), cmap = 'gray')
        plt.show()

    
    def normalize(self, images):
        return np.array([image/127.5-1 for image in images])
    

def read_data(data_dir, dataset = 'sr'):#, is_ycbcr = True):
    img_paths, images = [], []
    if dataset is 'sr':
        for root, dirs, files in os.walk(data_dir):
            if files:
                for f in files:
                    img_paths.append(os.path.join(root, f))
        for files in img_paths:                
            images.append(np.array(Image.open(files)))        

    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), data_dir)), dataset)
        data = glob.glob(os.path.join(data_dir, "*.png"))
        if not data:
            data = glob.glob(os.path.join(data_dir, "*.jpg"))
#        if is_ycbcr is True:
#            images_y, images_cb, images_cr  = [], [], []
#            for d in data:
#                img = Image.open(d)
#                img = img.convert('YCbCr')
#                y, cb, cr = img.split()
#                images_y.append(np.array(y))
#                images_cb.append(np.array(cb))
#                images_cr.append(np.array(cr))
#            return images_y, images_cb, images_cr
#        else:
        if len(data)>10000:
            data = data[:10000]
        images = [np.array(Image.open(d)) for d in data]
    return images

def prepare_data(images, size, sr_ratio = 1, is_rgb = True):
    hr_images, lr_images = [], []

    for im in images:
        h, w = im.shape[:2]
        if len(im.shape)<3:
            im = np.stack((im,)*3)
        if h > w:
            max_crop_size = w
        else:
            max_crop_size = h
        ims = crop_center(im, max_crop_size, max_crop_size) 
        hr_img = imresize(ims, (size*sr_ratio, size*sr_ratio))
        lr_img = imresize(hr_img, (1./sr_ratio), 'bicubic')
        hr_images.append(hr_img)
        lr_images.append(lr_img) 
                   
    hr = np.array(hr_images)
    lr = np.array(lr_images)
   
    return hr, lr
        
def crop(x, width = None, height=None):
    h, w, c = np.shape(x)
    cropped_x = []
    if width:
        i = int(round((w - width)/2.))
        cropped_x.append(x[:,i:i+width,:])
        cropped_x.append(x[:,0: width,:])
        cropped_x.append(x[:,w-width-1: w-1,:])
    else:        
        j = int(round((h - height)/2.))
        cropped_x.append(x[j:j+height,:,:])
        cropped_x.append(x[0:height,:,:])
        cropped_x.append(x[h-height-1:h-1,:,:])
    return cropped_x
    
def crop_center(img,cropx,cropy):
    if len(img.shape) == 3:
        y,x,_ = img.shape
    else:
        y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2) 
    if len(img.shape) == 3:
        return img[starty:starty+cropy,startx:startx+cropx,:]
    else:
        return img[starty:starty+cropy,startx:startx+cropx]