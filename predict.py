import numpy as np
import os
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from keras import backend as K
import cv2
from tifffile import imsave
from tensorflow.keras.models import load_model
import argparse
from utils import resize_to_512, resize_back
import cc3d
from keras.models import load_model
from utils import denoise
from keras import backend as K
from loss import dice_loss, binary_focalloss
from metrics import iou_score, f1_score, precision, recall


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  
  parser.add_argument("--input", help="test image")
  parser.add_argument("--model_dir", help="saved model directory")
  parser.add_argument("--model_name128", help="saved model 1")
  parser.add_argument("--patch_128_only", default = False, type=bool, help="use model with patch size 128 only")
  parser.add_argument("--model_name256", help="saved model 2")
  parser.add_argument("--patch_256_only", default= False, type=bool, help="use model with patch size 256 only")
  parser.add_argument("--both_model", default = False, type=bool, help="use both model versions")
  parser.add_argument("--denoise", default = False, help="whether to denoise the input image")
  args = parser.parse_args()
  return args

def morph_transformation(image, z):

  new = np.zeros((z, 512, 512), dtype='uint8')

  element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
  openingSize = 2
  k = 1
  it=1
  # Selecting a elliptical kernel
  element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
              (2 * openingSize + k, 2 * openingSize + k),
              (openingSize,openingSize))
  
  for i in range(z):
    imDilated = cv2.erode(image[i, :, :], element2)
    imClose = cv2.dilate(imDilated, element)
    new[i, :, :] = imClose
  
  imageMorphOpened = cv2.morphologyEx(new, cv2.MORPH_OPEN, 
                          element,iterations=it)  

  return imageMorphOpened

def get_prediction(model, image, patch_size, step):

  pred = np.zeros((32, 512, 512), dtype= "float32")
  prob = np.zeros((32, 512, 512), dtype= "float32")

  # Predict each 3D patch   
  p = (512-patch_size)//step + 1

  
  for i in range(p):
      
    for j in range(p):

      single_patch = image[:,i*step:i*step+patch_size, j*step:j*step+patch_size]
      single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
      single_patch_3ch_input = np.expand_dims(single_patch_3ch, axis=0)
      single_patch_prediction = model.predict(single_patch_3ch_input)
      
      patch_mask = np.reshape(single_patch_prediction, (32,patch_size,patch_size))
      patch_mask = (patch_mask - patch_mask.min())/(patch_mask.max()-patch_mask.min())

      prob[:, i*step:i*step+patch_size, j*step:j*step+patch_size] = np.maximum(pred[:, i*step:i*step+patch_size, j*step:j*step+patch_size],patch_mask)
      
      patch_mask = (patch_mask>0.5).astype(np.uint8)
      pred[:, i*step:i*step+patch_size, j*step:j*step+patch_size] = pred[:, i*step:i*step+patch_size, j*step:j*step+patch_size] + patch_mask
      pred = np.where(pred>0, 1, 0).astype(pred.dtype)
      
  return pred


def main():

  args = get_args()

  large_image = io.imread(os.path.join(os.getcwd(), args.input))
  if args.denoise:
    large_image = denoise(os.path.join(os.getcwd(), args.input))

  l_img = large_image

  large_image = (large_image - large_image.min())/(large_image.max()-large_image.min())
  large_image = resize_to_512(large_image)
  gamma = 1
  total_loss = dice_loss + (gamma * binary_focalloss)

  ### Loading the models ###
  if args.patch_128_only:
    my_model1 = load_model(os.path.join(args.model_dir, args.model_name128), custom_objects={"K": K, 'dice_loss_plus_1binary_focal_loss':total_loss, 'iou_score':iou_score, 'f1_score':f1_score, 'precision':precision, 'recall':recall}, compile=False)
    patch_size = 128
    step = 64
    predicted_image = get_prediction(my_model1, large_image, patch_size, step)

  if args.patch_256_only:
    my_model2 = load_model(os.path.join(args.model_dir + args.model_name256),custom_objects={"K": K, 'dice_loss_plus_1binary_focal_loss':total_loss, 'iou_score':iou_score, 'f1_score':f1_score, 'precision':precision, 'recall':recall}, compile=False)
    patch_size = 256
    step = 128
    predicted_image = get_prediction(my_model2, large_image, patch_size, step)

  if args.both_model:
    my_model1 = load_model(os.path.join(args.model_dir, args.model_name128), custom_objects={"K": K, 'dice_loss_plus_1binary_focal_loss':total_loss, 'iou_score':iou_score, 'f1_score':f1_score, 'precision':precision, 'recall':recall}, compile=False)
    patch_size = 128
    step = 64
    predicted1 = get_prediction(my_model1, large_image, patch_size, step)
    my_model2 = load_model(os.path.join(args.model_dir + args.model_name256),custom_objects={"K": K, 'dice_loss_plus_1binary_focal_loss':total_loss, 'iou_score':iou_score, 'f1_score':f1_score, 'precision':precision, 'recall':recall}, compile=False)
    patch_size = 256
    step = 128
    predicted2 = get_prediction(my_model2, large_image, patch_size, step)
    predicted_image = (predicted1 + predicted2) / 2
  
  ### Post-Processing ###

  z = l_img.shape[0]
  prediction = resize_back(predicted_image, z)
  imsave('prediction.tif', prediction)

  image_morphed = morph_transformation(prediction, z)
  labels, N = cc3d.connected_components(image_morphed, return_N=True)

  labels_out = cc3d.dust(
  labels, threshold=10000, 
  connectivity=6, in_place=False
  )
  stats = cc3d.statistics(labels_out)
  # SHOW MAX PROJECTION
  imsave('labels.tif', labels_out)
  IM_MAX= np.max(labels_out, axis=0)

  # MEASURE VOLUME

  img = (IM_MAX*255/6).astype(np.uint16)
  img = np.stack((img,)*3, axis=-1)

  for i in range(1,len(stats['centroids'])):
    j = stats['centroids']
    v = stats['voxel_counts']
    if not v[i]==0:
      
      cv2.putText(img, "{:.1f}".format(v[i]*0.02045), (int(j[i][2]), int(j[i][1])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (255, 0, 0), 1)
  imsave('result.tif', img)

if __name__ == "__main__":
    main()