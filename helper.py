SMOOTH = 1e-5
import tensorflow as tf
from tensorflow import keras

backend=keras.backend
layers=keras.layers
models=keras.models
utils=keras.utils

class KerasObject:
    _backend = keras.backend
    _models = keras.models
    _layers = keras.layers
    _utils = keras.utils

    def __init__(self, name=None):
        if (self.backend is None or
                self.utils is None or
                self.models is None or
                self.layers is None):
                                 
            raise RuntimeError('You cannot use `KerasObjects` with None submodules.')

        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name):
        self._name = name

    @classmethod
    def set_submodules(cls, backend, layers, models, utils):
        cls._backend = backend
        cls._layers = layers
        cls._models = models
        cls._utils = utils

    @property
    def submodules(self):
        return {
            'backend': self.backend,
            'layers': self.layers,
            'models': self.models,
            'utils': self.utils,
        }

    @property
    def backend(self):
        return self._backend

    @property
    def layers(self):
        return self._layers

    @property
    def models(self):
        return self._models

    @property
    def utils(self):
        return self._utils


class Metric(KerasObject):
    pass


class Loss(KerasObject):

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)


class MultipliedLoss(Loss):

    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{}({})'.format(multiplier, loss.__name__)
        else:
            name = '{}{}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, gt, pr):
        return self.multiplier * self.loss(gt, pr)


class SumOfLosses(Loss):

    def __init__(self, l1, l2):
        name = '{}_plus_{}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, gt, pr):
        return self.l1(gt, pr) + self.l2(gt, pr)

def _gather_channels(x, indexes, **kwargs):
    """Slice tensor along channels axis by given indexes"""

    if backend.image_data_format() == 'channels_last':
        x = backend.permute_dimensions(x, (4, 0, 1, 2, 3))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 2, 3, 4, 0))
    else:
        x = backend.permute_dimensions(x, (1, 0, 2, 3, 4))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3, 4))
    return x


def get_reduce_axes(per_image, **kwargs):

    axes = [1, 2, 3] if backend.image_data_format() == 'channels_last' else [2, 3, 4]
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(*xs, indexes=None, **kwargs):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
    return xs


def round_if_needed(x, threshold, **kwargs):
  
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx())
    return x


def average(x, per_image=False, class_weights=None, **kwargs):
 
    if per_image:
        x = backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return backend.mean(x)

def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None,
            **kwargs):

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # calculate score
    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    fn = backend.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


# ----------------------------------------------------------------
#   Loss Functions
# ----------------------------------------------------------------

def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None, **kwargs):


    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # scale predictions so that the class probas of each sample sum to 1
    axis = 4 if backend.image_data_format() == 'channels_last' else 1
    pr /= backend.sum(pr, axis=axis, keepdims=True)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1 - backend.epsilon())

    # calculate loss
    output = gt * backend.log(pr) * class_weights
    return - backend.mean(output)


def binary_crossentropy(gt, pr, **kwargs):
 
    return backend.mean(backend.binary_crossentropy(gt, pr))


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None, **kwargs):
   

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate focal loss
    loss = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))

    return backend.mean(loss)


def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25, **kwargs):
 

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    loss_1 = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))
    loss_0 = - (1 - gt) * ((1 - alpha) * backend.pow((pr), gamma) * backend.log(1 - pr))
    loss = backend.mean(loss_0 + loss_1)
    return loss