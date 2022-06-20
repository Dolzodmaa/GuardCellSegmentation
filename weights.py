from tensorflow import keras

backend=keras.backend
layers=keras.layers
models=keras.models
utils=keras.utils

def _find_weights(model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, WEIGHTS_COLLECTION))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w


def load_model_weights(model, model_name, dataset, classes, include_top, **kwargs):
    
    weights = _find_weights(model_name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))

        weights_path = utils.get_file(
            weights['name'],
            weights['url'],
            cache_subdir='models',
            md5_hash=weights['md5']
        )

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))

__all__ = ['load_model_weights']

WEIGHTS_COLLECTION = [

    # densenet
    {
        'model': 'densenet',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/densenet121_inp_channel_3_tch_0_top_False.h5',
        'name': 'densenet121_inp_channel_3_tch_0_top_False.h5',
        'md5': '743ea52b43c19000d9c4dcd328fd3f9d',
    },
    # densenet
    {
        'model': 'densenet',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/densenet121_inp_channel_3_tch_0_top_True.h5',
        'name': 'densenet121_inp_channel_3_tch_0_top_True.h5',
        'md5': 'c9dec0d11eda5fb3ca85369849dbdc6c',
    },
   
    
  
   
]