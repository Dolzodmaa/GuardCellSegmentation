
import tensorflow as tf
import keras
import numpy as np
import os
import segmentation_models_3D as sm
from skimage import io, img_as_float, img_as_ubyte
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from tensorflow.keras.layers import Input, Conv3D,Conv2D, SpatialDropout3D,SpatialDropout2D, Conv3DTranspose, Conv2DTranspose,MaxPooling3D,MaxPooling2D, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import cv2
from skimage.transform import rescale, resize
from PIL import Image
from tifffile import imsave, imread
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from skimage.restoration import denoise_nl_means, estimate_sigma
#tensorflow.from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

def dataset_loader(img_dir, mask_dir):


##Convert grey image to 3 channels by copying channel 3 times.
#We do this as our unet model expects 3 channel input. 

    train_img_dir = os.listdir(img_dir_slice)
    train_mask_dir = os.listdir(label_dir_slice)
    name = {'13.tif':'0.tif', '14.tif':'1.tif', '7.tif':'2.tif', '8.tif':'3.tif', 'N-1.tif':'4.tif', 'N-11.tif':'5.tif',
            'N-2.tif':'6.tif', 'N-5.tif':'7.tif', 'N-9.tif':'8.tif'}

    input_img, input_mask = [], []
    for i, img in enumerate(sorted(train_img_dir)):
    print(img)
    image = imread(img_dir_slice+img)
    input_img.append(image)
    #image = resize_to_512(image)
    #sample = expand_dims(image, 0)
    #it = datagen.flow(sample, batch_size=1)
    #image = repeat_to_512(image, image.shape[0], 32)
    #print(image.shape)
    #print(image.max())

    #img_patches = make_patches(image) 
    #print(img_patches.shape)
    """if i==0:
        input_img = img_patches
    else:
        input_img = np.concatenate((input_img, img_patches), axis = 0)
    aug_img = io.imread(img_dir+'0'+img)
    aug_img = resize_to_512(aug_img)
    aug_img = aug_img[:, 100:356, 100:356]
    aug_img /= aug_img.max()
    aug_img_patches = make_patches(aug_img)
    input_img = np.concatenate((input_img, aug_img_patches), axis = 0)"""

    mask = io.imread(label_dir_slice+img)
    input_mask.append(mask)
    #mask = resize_to_512(mask)
    #mask = resize_to_512(mask)
    #mask = pad_to_full(mask, mask.shape[0])
    #mask_patches = make_patches(mask)
    """if i==0:
        input_mask = mask_patches
    else:
        input_mask = np.concatenate((input_mask, mask_patches), axis = 0) 
    aug_mask = io.imread(label_dir+'0'+'label-'+img)
    aug_mask = resize_to_512(aug_mask)
    aug_mask = aug_mask[:, 100:356, 100:356]
    aug_mask /= aug_mask.max()
    aug_mask_patches = make_patches(aug_mask)
    input_mask = np.concatenate((input_mask, aug_mask_patches), axis = 0)
    for i in range(1):
        gen_img = np.squeeze(io.imread(img_dir+'g'+str(i)+'_img'+ name[img]), axis=0)
        print(gen_img.shape)
        gen_img = repeat_to_512(gen_img, gen_img.shape[0], 32)
        gen_img_patches = make_patches(gen_img)
        input_img = np.concatenate((input_img, gen_img_patches), axis = 0)

        gen_mask = np.squeeze(io.imread(label_dir+'g'+str(i)+'_label'+ name[img]), axis=0)
        gen_mask = repeat_to_512(gen_mask, gen_mask.shape[0], 32)
        gen_mask_patches = make_patches(gen_mask)
        input_mask = np.concatenate((input_mask, gen_mask_patches), axis = 0)
    """

    input_img = np.array(input_img).astype(np.float32)
    input_mask = np.array(input_mask).astype(np.float32)

    train_img = np.stack((input_img,)*3, axis=-1)
    train_mask = np.expand_dims(input_mask, axis=3)

    del input_img
    del input_mask

    X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask, test_size = 0.20, random_state = 0, shuffle=True)

    return X_train, X_test, y_train, y_test
