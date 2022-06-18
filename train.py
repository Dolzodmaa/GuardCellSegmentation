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
import math
from skimage.transform import rescale, resize
from PIL import Image
from tifffile import imsave
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from skimage.restoration import denoise_nl_means, estimate_sigma
#tensorflow.from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model
from loss import dice_loss, binary_focal_loss


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
   
    parser.add_argument("--log_dir", help="directory for logging.")
    parser.add_argument("--learning_rate", default = 0.001)
    parser.add_argument("--epoch", default = 100, help="number of epochs to train")
    parser.add_argument("--")

    
    args = parser.parse_args()
    return args


#Define parameters for our model.




def main():
    args = get_args()
   
    encoder_weights = 'imagenet'
    BACKBONE = 'densenet121' 
    activation = 'sigmoid'
    patch_size = 4
    n_classes = 1
    channels=3

    LR = 0.001
    optim = tf.keras.optimizers.Adam(LR)
    gamma = 1
    total_loss = dice_loss + (gamma * binary_focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model = sm.AttnUnet(BACKBONE, classes=n_classes, 
                    input_shape=(32, 128, 128, channels), 
                    encoder_weights='imagenet',
                    activation=activation,
                    dropout=0.15
                    )
    model.compile(optimizer = optim, loss=total_loss, metrics=metrics)


    initial_learning_rate = 0.001
    def lr_exp_decay(epoch, lr):
        k = 0.01
        return initial_learning_rate * math.exp(-k*epoch)
    early_stopping = EarlyStopping(monitor='val_loss', 
        patience=10, 
        min_delta=0, 
        mode='auto')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1,   
        patience=2, 
        min_lr=0.0001,
        verbose=2
    )
    checkpoint_filepath = '/content/drive/MyDrive/Guard_cell_data/Models/'
    filepath = 'my_best_model.epoch{epoch:02d}-loss{val_iou_score:.2f}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath,filepath),
        save_weights_only=True,
        monitor='val_iou_score',
        mode='max',
        save_best_only=True,
        verbose = 1)

    history=model.fit(X_train_prep, 
            y_train,
            batch_size=8, 
            epochs=100,
            verbose=1,
            validation_data=(X_test_prep, y_test),
            callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1), early_stopping, model_checkpoint_callback])



if __name__ == "__main__":
    main()