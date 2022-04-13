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