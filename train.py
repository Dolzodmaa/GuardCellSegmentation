

import tensorflow as tf
import numpy as np
import os
import pickle
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import math
import functools
from tifffile import imsave
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from skimage.restoration import denoise_nl_means, estimate_sigma
from tensorflow.keras.models import load_model
from loss import dice_loss, binary_focalloss
from model import Attention_UNet
from dataset import dataset_loader
import argparse
from tensorflow import keras
import helper
helper.KerasObject.set_submodules(
        backend= keras.backend,
        layers= keras.layers,
        models= keras.models,
        utils= keras.utils,
    )
from metrics import iou_score, f1_score, precision, recall


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
   
    parser.add_argument("--log_dir", help="directory for logging.")
    parser.add_argument("--img_dir", help="training image location")
    parser.add_argument("--mask_dir", help="training label location")
    parser.add_argument("--patch_shape", default = 256, type=int, help="shape of patches")
    parser.add_argument("--learning_rate", default = 0.0001)
    parser.add_argument("--patch_step", default = 128, type=int, help="step size for patch windows: 64 for patch size 128, 128 for patch size 256")
    parser.add_argument("--epochs", default = 100, type =int, help="number of epochs to train")
    parser.add_argument("--batch_size", default = 4, help="batch size for training")
    parser.add_argument("--gamma_total_loss", default = 3, help="weight for focal loss")
    parser.add_argument("--dropout_rate", default = 0.15, help="dropout rate")
    parser.add_argument("--load_model", default = False, help="train from the saved model")
    parser.add_argument("--model_dir", help="saved model directory")
    args = parser.parse_args()
    return args
    
def inject_global_submodules(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        return func(*args, **kwargs)

    return wrapper

def main():
    helper.KerasObject.set_submodules(
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils,
    )
    args = get_args()
   
    BACKBONE = 'densenet' 
    activation = 'sigmoid'
    LR = args.learning_rate
    optim = tf.keras.optimizers.Adam(LR)
    gamma = args.gamma_total_loss
    total_loss = dice_loss + (gamma * binary_focalloss)
    metrics = [iou_score, f1_score, precision, recall]

    model = Attention_UNet(BACKBONE, classes=1, 
                    input_shape=(32, args.patch_shape, args.patch_shape, 3), 
                    encoder_weights='imagenet',
                    activation=activation,
                    dropout=args.dropout_rate
                    )

    model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
   
    early_stopping = EarlyStopping(monitor='val_iou_score', 
        patience=20, 
        min_delta=0, 
        mode='auto')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1,   
        patience=10, 
        min_lr=0.00001,
        verbose=2
    )
    checkpoint_filepath = args.log_dir
    filename = 'p_256_model.epoch{epoch:02d}-loss{val_iou_score:.2f}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath,filename),
        save_weights_only=False,
        monitor='val_iou_score',
        mode='max',
        save_best_only=True,
        verbose = 1)
 
    X_train, X_test, y_train, y_test = dataset_loader(args.img_dir, args.mask_dir, args.patch_shape, args.patch_step)
    history = model.fit(X_train, 
            y_train,
            batch_size=args.batch_size, 
            epochs=args.epochs,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[ reduce_lr, model_checkpoint_callback])

    with open('/content/drive/MyDrive/Guard_cell_data/trainHistoryDict_256', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == "__main__":
    main()