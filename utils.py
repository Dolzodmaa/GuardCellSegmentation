from keras_applications import get_submodules_from_kwargs
from patchify import patchify, unpatchify
import numpy as np
import cv2
from tensorflow import keras
import functools
from skimage import io, img_as_float, morphology
from skimage.restoration import denoise_nl_means, estimate_sigma


def set_trainable(model, recompile=True, **kwargs):
   
    for layer in model.layers:
        layer.trainable = True

    if recompile:
        model.compile(
            model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
            loss_weights=model.loss_weights,
            sample_weight_mode=model.sample_weight_mode,
            weighted_metrics=model.weighted_metrics,
        )


def set_regularization(
        model,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        beta_regularizer=None,
        gamma_regularizer=None,
        **kwargs
):
   
    _, _, models, _ = get_submodules_from_kwargs(kwargs)

    for layer in model.layers:
        # set kernel_regularizer
        if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer
        # set bias_regularizer
        if bias_regularizer is not None and hasattr(layer, 'bias_regularizer'):
            layer.bias_regularizer = bias_regularizer
        # set activity_regularizer
        if activity_regularizer is not None and hasattr(layer, 'activity_regularizer'):
            layer.activity_regularizer = activity_regularizer

        # set beta and gamma of BN layer
        if beta_regularizer is not None and hasattr(layer, 'beta_regularizer'):
            layer.beta_regularizer = beta_regularizer

        if gamma_regularizer is not None and hasattr(layer, 'gamma_regularizer'):
            layer.gamma_regularizer = gamma_regularizer

    out = models.model_from_json(model.to_json())
    out.set_weights(model.get_weights())

    return out


def freeze_model(model, **kwargs):
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return

def resize_to_512(image):
  img_stack_sm = np.zeros((32, 512, 512), dtype="float32")

  for idx in range(512):
      img = image[:, :, idx]
      img_sm = cv2.resize(img, (512, 32), interpolation= cv2.INTER_LINEAR)
      img_stack_sm[:, :, idx] = img_sm
      
  return img_stack_sm

def make_patches(image, patch_shape, step):
  img_patches = patchify(image, (32, patch_shape, patch_shape), step=step)
  input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
  return input_img

def denoise(img_path):

  image = io.imread(img_path)
  image_float = img_as_float(image)
  sigma_est = np.mean(estimate_sigma(image,multichannel=False))
  nlm = denoise_nl_means(image, h=1.15*sigma_est, patch_size=5, patch_distance=3)
  nlm_uint = nlm.astype('uint16')
  nlm_uint = nlm_uint/nlm_uint.max()
  return nlm_uint

def resize_back(image, dim):
  img_stack_sm = np.zeros((dim, 512, 512), dtype="uint8")

  for idx in range(512):
      img = image[:, :, idx]
      img_sm = cv2.resize(img, (512, dim), interpolation= cv2.INTER_LINEAR)
      img_stack_sm[:, :, idx] = img_sm
      
  return img_stack_sm