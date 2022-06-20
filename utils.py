""" Utility functions for segmentation models """

from keras_applications import get_submodules_from_kwargs
from patchify import patchify, unpatchify
import numpy as np
import cv2
import keras
import functools

def inject_global_submodules(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        return func(*args, **kwargs)

    return wrapper

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


@inject_global_submodules
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
    """Set all layers non trainable, excluding BatchNormalization layers"""
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

def make_patches(image):
  img_patches = patchify(image, (32, 256, 256), step=64)
  input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
  return input_img

