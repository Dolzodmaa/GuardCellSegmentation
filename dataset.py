
import tensorflow as tf
import numpy as np
import os
from skimage import io, img_as_float, img_as_ubyte
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from sklearn.model_selection import train_test_split
import cv2
from skimage.transform import rescale, resize
from PIL import Image
from tifffile import imsave, imread
from skimage.restoration import denoise_nl_means, estimate_sigma
from utils import resize_to_512, make_patches

def dataset_loader(img_dir, mask_dir):

    train_img_dir = os.listdir(img_dir)
    train_mask_dir = os.listdir(mask_dir)

    input_img, input_mask = [], []
    for i, img in enumerate(sorted(train_img_dir)):
   
        image = imread(img_dir+img)
        image = resize_to_512(image)
        input_img.append(image)
        img_patches = make_patches(image) 
        print(img_patches.shape)
        if i==0:
            input_img = img_patches
        else:
            input_img = np.concatenate((input_img, img_patches), axis = 0)
    
        mask = io.imread(mask_dir+img)
        input_mask.append(mask)
        mask = resize_to_512(mask)
        mask_patches = make_patches(mask)
        if i==0:
            input_mask = mask_patches
        else:
            input_mask = np.concatenate((input_mask, mask_patches), axis = 0) 
    

    input_img = np.array(input_img).astype(np.float32)
    input_mask = np.array(input_mask).astype(np.float32)

    train_img = np.stack((input_img,)*3, axis=-1)
    train_mask = np.expand_dims(input_mask, axis=3)

    del input_img
    del input_mask

    X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask, test_size = 0.20, random_state = 0, shuffle=True)

    return X_train, X_test, y_train, y_test
