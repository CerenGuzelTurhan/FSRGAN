import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import scipy
from PIL import Image
from scipy.misc import imresize
import pickle
import json
import shutil
import math


def read_dataset(data_dir = './data/MNIST/', one_hot = False):
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
        
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0)
       
    seed = np.random.randint(0,len(X))
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    if one_hot:    
        y_vec = np.zeros((len(y), None), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0
            
        return X/255.,y_vec
    else:
        return X/255., y
        
def plot(samples, save = False, name = None):

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
    if save:
        samples_file = 'samples/' 
        fig.savefig(os.path.join(samples_file, name, '.png'))

def get_random_image(dataset, name, n = 1):
    if name is "mnist":
        x = dataset.next_batch(n)[0]
    else:      
        x = dataset[np.random.randint(0,len(dataset), size = n)]
    return x

def get_random_specific_image(dataset, labels, name, mylabel, n=1):
    if name is "mnist":
        x,y = dataset.next_batch(n, True)
        while int(y) is not mylabel:
            x,y = dataset.next_batch(n, True)
        image = x[n-1]
    else:
        i = np.random.randint(0,len(dataset))
        img, y = dataset[i], labels[i]
        image = np.zeros((n, dataset.shape[1], dataset.shape[2], dataset.shape[3]))
        for j in range(1,n):
            while int(y) is not mylabel:
                i = np.random.randint(0,len(dataset))
                img, y = dataset[i], labels[i]
            image[i]=img
    return image
    
def show_image(image_data):
    plt.subplot(1,1,1)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = image_data.shape[2]
    if c_dim > 1:
        plt.imshow(image_data, interpolation = 'nearest')
    else:
        plt.imshow(image_data.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()
    
def save_image(image_data, filename):
    img_data = np.array(1-image_data)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = image_data.shape[2]
    if c_dim > 1:
      img_data=  img_data.reshape((y_dim, x_dim, c_dim))
      img_data = np.array((255.0 / img_data.max() * (img_data - img_data.min())).astype(np.uint8))
    else:
      img_data = np.array(img_data.reshape((y_dim, x_dim))*255.0, dtype=np.uint8)
    im = Image.fromarray(img_data)
    im.save(filename)
    
def saveimg2mat(images, filedir):
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    scipy.io.savemat(filedir+ 'test.mat', {'images': images})
    
def imread(path):
    img = Image.open(path)
    gray = img.convert('L')
    gray_img = np.asarray(gray)
    return np.array(255-gray_img)

def resize_width(image, width=64.):
    h, w = np.shape(image)[:2]
    return scipy.misc.imresize(image,[int((float(h)/w)*width),width])
        
def center_crop(x, height=64):
    h = np.shape(x)[0]
    j = int(round((h - height)/2.))
    return x[j:j+height]

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
def get_image_withcrop(image_path, width=64, height=64):
    return center_crop(resize_width(imread(image_path), width = width),height=height)     

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

def prepare_data(images, size, sr_ratio = 1, is_rgb = True, is_data_aug = True, is_shuffle = False, is_normalized = True, is_ycbcr=True):
    hr_images, lr_images = [], []
    if is_data_aug is True:
        for im in images:
            h, w = im.shape[:2]
            if len(im.shape)<3:
                im = np.stack((im,)*3)
            if h > w:
                max_crop_size = w
                ims = crop(im, height = max_crop_size)
            elif h < w:
                max_crop_size = h
                ims = crop(im, width = max_crop_size)
            else:
                ims = []
                hr_img = imresize(im, (size*sr_ratio, size*sr_ratio))
                lr_img = imresize(hr_img, (1./sr_ratio), 'bicubic')
                hr_images.append(hr_img)
                lr_images.append(lr_img) 
            for i in ims:
                hr_img = imresize(i, (size*sr_ratio, size*sr_ratio))
                lr_img = imresize(hr_img, (1./sr_ratio), 'bicubic')
                hr_images.append(hr_img)
                lr_images.append(lr_img) 
    else:
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
            
    if is_ycbcr is True:
        hr, _, _ = rgb2ycbcr(np.array(hr_images))
        lr,cb, cr = rgb2ycbcr(np.array(lr_images))
        #imgs = imgs/255.
        #imgs_gnd = imgs_gnd/255.   
    else:
        hr = np.array(hr_images)
        lr = np.array(lr_images)
    if is_normalized is True:
        hr = hr/255.
        lr = lr/255.
        if is_rgb is False:
            hr = np.dot(hr[...,:3], [0.299, 0.587, 0.114]).reshape(len(hr), int(size*sr_ratio), int(size*sr_ratio), 1)
            lr = np.dot(lr[...,:3], [0.299, 0.587, 0.114]).reshape(len(lr), size, size, 1)
    if is_shuffle is True:
        seed = np.random.randint(0,len(hr)-1)
        np.random.seed(seed)
        np.random.shuffle(hr)
        np.random.seed(seed)
        np.random.shuffle(lr)
    if is_ycbcr is True:
        return hr, lr, cb, cr
    else: 
        return hr, lr
    
def prepare_cropped_data(lr_size = None, is_ycbcr = True):
    with open("params.json.txt", 'r') as f:
        params = json.load(f)
    params['hr_stride'] = params['lr_stride'] * params['ratio']
    if not lr_size:
        params['hr_size'] = params['lr_size'] * params['ratio']
    else:
        params['hr_size'] = lr_size * params['ratio']
    params['padding'] = int((params['hr_size'] - params['lr_size']) /2)
    targetPatch = params['hr_size']
    inputPatch = int(targetPatch / params['ratio'])
    image_dirs = [params['training_image_dir'], params['validation_image_dir'], params['test_image_dir']]
# =============================================================================
#     hr_start_idx = int(params['ratio'] * params['edge'] / 2)
#     hr_end_idx = int(hr_start_idx + (params['lr_size'] - params['edge']) * params['ratio'])
#     lr_start_idx = int(params['edge'] / 2)
#     lr_end_idx = int(lr_start_idx + (params['lr_size'] - params['edge']))
#     sub_hr_size = (params['lr_size'] - params['edge']) * params['ratio']
# =============================================================================
    hr_imgs = []
    lr_imgs = []
    
    for dir_idx, image_dir in enumerate(image_dirs):
        for root, dirnames, filenames in os.walk(image_dir + "lr_x8/"):
            for filename in filenames:
                lr_path = os.path.join(root, filename)
                hr_path = image_dir + "hr/" + filename.split('x')[0] +'.png'
                lr_image = np.array(Image.open(lr_path))
                hr_image = np.array(Image.open(hr_path))
                h, w, c = hr_image.shape
                h_lr, w_lr = math.floor(h/ params['ratio']), math.floor(w/ params['ratio'])
                h_hr, w_hr = h_lr * params['ratio'], w_lr * params['ratio']

                #--Generate patches for training      
                #ix = np.random.randint(1, w_lr - inputPatch + 1)
                #iy = np.random.randint(1, h_lr - inputPatch + 1)
                for ix in range(params['edge'], w_lr - inputPatch - params['edge']+ 1, params['hr_stride']):
                    for iy in range(params['edge'], h_lr - inputPatch - params['edge'] + 1, params['hr_stride']):
                        tx = params['ratio'] * (ix - 1) + 1
                        ty = params['ratio'] * (iy - 1) + 1
                        sub_lr_image = lr_image[iy: iy + inputPatch, ix: ix + inputPatch]
                        sub_hr_image = hr_image[ty: ty + targetPatch, tx: tx + targetPatch]
                        hr_imgs.append(sub_hr_image)
                        lr_imgs.append(sub_lr_image)
    
    # convert to Ycbcr color space
    if is_ycbcr is True:
        lr, cb, cr = rgb2ycbcr(np.array(lr_imgs))
        hr, _, _ = rgb2ycbcr(np.array(hr_imgs))   
        return hr/255., lr/255., cb, cr, np.array(hr_imgs), np.array(lr_imgs)
    else:
        return np.array(hr_imgs)/255., np.array(lr_imgs)/255.
        

   
def rgb2ycbcr(images):
    imgs_y, imgs_cb, imgs_cr = [], [], []
# =============================================================================
#     if np.max(images)<= 1.0:
#         images = images * 255
# =============================================================================
    for i in range(len(images)):        
        y, cb, cr = Image.fromarray(np.uint8(images[i])).convert('YCbCr').split()
        imgs_y.append(np.asarray(y))
        imgs_cb.append(np.asarray(cb))
        imgs_cr.append(np.asarray(cr))
    return np.array(imgs_y), np.array(imgs_cb), np.array(imgs_cr)

def ycbcr2rgb(imgs_y, imgs_cb, imgs_cr):
    out_img = []
    if np.mean(imgs_y) <= 1.0:
        imgs_y = ((imgs_y - np.min(imgs_y)) * 255) / (np.max(imgs_y) - np.min(imgs_y))
        #imgs_y = imgs_y * 255
    for i in range(len(imgs_y)):
        out_img_y = Image.fromarray(np.uint8(imgs_y[i]), mode='L')
        out_img_cb = Image.fromarray(imgs_cb[i]).resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = Image.fromarray(imgs_cr[i]).resize(out_img_y.size, Image.BICUBIC)
        out_img.append(np.asarray(Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')))
                                        
    return np.array(out_img)
        
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)

    return dict   

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def load_databatch(data_folder, idx, img_size=32, is_rand = True, is_shuffle =False):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    #y = d['labels']
    #mean_image = d['mean']

    #x = x/np.float32(255)
    #mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    #y = [i-1 for i in y]
    #data_size = x.shape[0]
    if is_rand is True:
        x_train = x[:120000]
        #Y_train = y[0:data_size]
    else:
        x_train = x[:10000]
    #x -= mean_image

    img_size2 = img_size * img_size

    x_train = np.dstack((x_train[:, :img_size2], x_train[:, img_size2:2*img_size2], x_train[:, 2*img_size2:]))
    x_train = x_train.reshape((x_train.shape[0], img_size, img_size, 3))#.transpose(0, 3, 1, 2)
    if is_shuffle is True:
        seed = 120
        np.random.seed(seed)
        np.random.shuffle(x_train)
    # create mirrored images

# =============================================================================
#     X_train_flip = X_train[:, :, :, ::-1]
#     Y_train_flip = Y_train
#     X_train = np.concatenate((X_train, X_train_flip), axis=0)
#     Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)
# =============================================================================

    return x_train

def loadImagenet(data_folder, org_size, sr_ratio, i = None, is_normalized = True, is_ycbcr = False):

    if i:
        HR_train = load_databatch(data_folder + '/hr', i, org_size, is_rand = False, is_shuffle = False)
        LR_train = np.array([imresize(HR_train[i], (1./sr_ratio), 'bicubic') for i in range(len(HR_train))])       
    else:
        HR_train = []
        LR_train = []
        for i in range(1, 10):
            HR = load_databatch(data_folder + '/hr', i, org_size) 
            HR_train.append(HR)
           # LR_train.append(load_databatch(data_folder + '/lr/x' + str(sr_ratio), i, int(org_size/sr_ratio)))
            LR_train.append(imresize(HR, (1./sr_ratio), 'bicubic'))
        HR_train = np.array(HR_train)
        LR_train = np.array(LR_train)
    if is_ycbcr is True:
        HR_train, _, _ = rgb2ycbcr(HR_train)
        LR_train,cb, cr = rgb2ycbcr(LR_train)
    if is_normalized is True:
        HR_train = HR_train/np.float32(255)
        LR_train = LR_train/np.float32(255)
    if is_ycbcr is True:
        return HR_train, LR_train, cb, cr
    else:
        return HR_train, LR_train

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