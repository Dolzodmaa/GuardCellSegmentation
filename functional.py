def _gather_channels(x, indexes, **kwargs):
    """Slice tensor along channels axis by given indexes"""
    backend = kwargs['backend']
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
    backend = kwargs['backend']
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
    backend = kwargs['backend']
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx())
    return x


def average(x, per_image=False, class_weights=None, **kwargs):
    backend = kwargs['backend']
    if per_image:
        x = backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return backend.mean(x)


# ----------------------------------------------------------------
#   Metric Functions
# ----------------------------------------------------------------

def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):


    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # score calculation
    intersection = backend.sum(gt * pr, axis=axes)
    union = backend.sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None,
            **kwargs):
   

    backend = kwargs['backend']

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

    backend = kwargs['backend']

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

    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    tp = backend.sum(gt * pr, axis=axes)
    fn = backend.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


# ----------------------------------------------------------------
#   Loss Functions
# ----------------------------------------------------------------

def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None, **kwargs):
    backend = kwargs['backend']

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
    backend = kwargs['backend']
    return backend.mean(backend.binary_crossentropy(gt, pr))


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None, **kwargs):


    backend = kwargs['backend']
    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate focal loss
    loss = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))

    return backend.mean(loss)


def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25, **kwargs):

    backend = kwargs['backend']

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    loss_1 = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))
    loss_0 = - (1 - gt) * ((1 - alpha) * backend.pow((pr), gamma) * backend.log(1 - pr))
    loss = backend.mean(loss_0 + loss_1)
    return loss