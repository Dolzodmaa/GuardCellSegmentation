from helper import gather_channels, round_if_needed, get_reduce_axes, average
from tensorflow import keras


backend=keras.backend
layers=keras.layers
models=keras.models
utils=keras.utils
SMOOTH = 1e-5


def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
  

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # score calculation
    intersection = backend.sum(gt * pr, axis=axes)
    union = backend.sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def f1_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None,
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


def precision(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):

   

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # score calculation
    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    
    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def recall(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
  
    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    tp = backend.sum(gt * pr, axis=axes)
    fn = backend.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score