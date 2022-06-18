import functools
import keras_applications as ka
from densenet import densenet as dn
import copy

class Model:
    _models = {
        'densenet': [dn.DenseNet, dn.preprocess_input],
    }

    @property
    def models(self):
        return self._models

    def models_names(self):
        return list(self.models.keys())

    @staticmethod
    def get_kwargs():
        return {}

    def inject_submodules(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            modules_kwargs = self.get_kwargs()
            new_kwargs = dict(list(kwargs.items()) + list(modules_kwargs.items()))
            return func(*args, **new_kwargs)

        return wrapper

    def get(self, name):
        
        model_fn, preprocess_input = self.models[name]
        model_fn = self.inject_submodules(model_fn)
        preprocess_input = self.inject_submodules(preprocess_input)
        return model_fn, preprocess_input

class Backbone(Model):
    _default_feature_layers = {
        # DenseNet
        'densenet': (311, 139, 51, 4),
    }

    @property
    def models(self):
        all_models = copy.copy(self._models)
        all_models.update(self._models_update)
        for k in self._models_delete:
            del all_models[k]
        return all_models

    def get_backbone(self, name, *args, **kwargs):
        model_fn, _ = self.get(name)
        model = model_fn(*args, **kwargs)
        return model

    def get_feature_layers(self, name, n=5):
        return self._default_feature_layers[name][:n]

    def get_preprocessing(self, name):
        return self.get(name)[1]


backbone = Backbone()