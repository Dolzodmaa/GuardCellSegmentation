import tensorflow as tf
import keras
import numpy as np
import os
import segmentation_models_3D as sm
from skimage import io, img_as_float, img_as_ubyte, morphology
#from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from skimage.transform import rescale, resize
from PIL import Image
from tifffile import imsave
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from skimage.restoration import denoise_nl_means, estimate_sigma
#tensorflow.from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model

from utils import resize_to_512, make_patches, resize_back

#Load the pretrained model for testing and predictions. 
from keras.models import load_model
my_model1 = load_model(latest_85, compile=False)
my_model2 = load_model(model_88, compile=False)
#model = load_model(unet_2d, compile=False)


#img_path = '/content/drive/MyDrive/Guard_cell_data/Training/new_img_test/f23a.tif'
large_image = io.imread('/content/drive/MyDrive/Guard_cell_data/Training/new_img_test/f4a.tif')
#large_image = denoise(img_path)
l_img = large_image
#large_image = repeat_to_512(large_image, large_image.shape[0])
large_image = (large_image - large_image.min())/(large_image.max()-large_image.min())
print(large_image.max())
print(large_image.min())
large_image = resize_to_512(large_image)
step = 32
patch_size = 256
#patches = patchify(large_image, (32, patch_size, patch_size), step=64)  
#print(large_image.shape)
#print(patches.shape)
pred = np.zeros((32, patch_size, patch_size), dtype= "float32")
pred2 = np.zeros((32, 512, 512), dtype= "float32")
prob = np.zeros((32, 512, 512), dtype= "float32")
# Predict each 3D patch   
predicted_patches = []
p1 = (512-patch_size)//step + 1
for i in range(p1):
  #print(i)
  for j in range(p1):
    #print(j)
    single_patch = large_image[:,i*step:i*step+patch_size, j*step:j*step+patch_size]
    #print('patch', single_patch.shape)
    single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
    single_patch_3ch_input = np.expand_dims(single_patch_3ch, axis=0)
    single_patch_prediction1 = my_model1.predict(single_patch_3ch_input)
    
    patch_mask1 = np.reshape(single_patch_prediction1, (32,patch_size,patch_size))

    prob[:, i*step:i*step+patch_size, j*step:j*step+patch_size] = np.maximum(pred2[:, i*step:i*step+patch_size, j*step:j*step+patch_size],patch_mask1)
    #patch_mask = np.sum(patch_mask1, patch_mask2)
    patch_mask1 = (patch_mask1>0.0001).astype(np.uint8)
    #print(patch_mask.shape)
    #predicted_patches.append(patch_mask)
    
    pred2[:, i*step:i*step+patch_size, j*step:j*step+patch_size] = pred2[:, i*step:i*step+patch_size, j*step:j*step+patch_size] + patch_mask1
    pred2 = np.where(pred2>0, 1, 0).astype(pred2.dtype)

step = 32
patch_size = 128
p2 = (512-patch_size)//step + 1


for i in range(p2):
  #print(i)
  for j in range(p2):
    #print(j)
    single_patch = large_image[:,i*step:i*step+patch_size, j*step:j*step+patch_size]
    #print('patch', single_patch.shape)
    single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
    single_patch_3ch_input = np.expand_dims(single_patch_3ch, axis=0)
    single_patch_prediction2 = my_model2.predict(single_patch_3ch_input)
    
    patch_mask2 = np.reshape(single_patch_prediction2, (32,patch_size,patch_size))

    prob[:, i*step:i*step+patch_size, j*step:j*step+patch_size] = np.maximum(pred2[:, i*step:i*step+patch_size, j*step:j*step+patch_size],patch_mask2)
    #patch_mask = np.sum(patch_mask1, patch_mask2)
    patch_mask2 = (patch_mask2>0.0001).astype(np.uint8)
    #print(patch_mask.shape)
    #predicted_patches.append(patch_mask)
    
    pred2[:, i*step:i*step+patch_size, j*step:j*step+patch_size] = pred2[:, i*step:i*step+patch_size, j*step:j*step+patch_size] + patch_mask2
    pred2 = np.where(pred2>0, 1, 0).astype(pred2.dtype)



z = l_img.shape[0]

test1 = resize_back(pred2,z)
test1.shape
pro = resize_back(prob, z)


def morph_transformation(image):

	new = np.zeros((z, 512, 512), dtype='uint8')

	element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
	openingSize = 2
	k = 1
	it=1
	# Selecting a elliptical kernel
	element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
	            (2 * openingSize + k, 2 * openingSize + k),
	            (openingSize,openingSize))
	black_pixels = np.where(image)
	for i in range(image[i, :, :], element2)
	  imClose = cv2.dilate(imDilated, element)
	  new[i, :, :] = imClose
	
	imageMorphOpened = cv2.morphologyEx(new, cv2.MORPH_OPEN, 
	                        element,iterations=it)  

	return imageMorphOpened