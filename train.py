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
from model import Attention_UNet
from dataset import dataset_loader
from metrics import iou_score, f1_score, precision, recall

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
   
    parser.add_argument("--log_dir", help="directory for logging.")
    parser.add_argument("--img_dir", help="training image location")
    parser.add_argument("--mask_dir", help="training label location")
    parser.add_argument("--learning_rate", default = 0.001)
    parser.add_argument("--epoch", default = 100, help="number of epochs to train")
    parser.add_argument("--batch_size", default = 4, help="batch size for training")
    parser.add_argument("--gamma_total_loss", default = 1, help="weight for focal loss")
    parser.add_argument("--dropout_rate", default = 0.15, help="dropout rate")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
   
    BACKBONE = 'densenet' 
    activation = 'sigmoid'
    LR = args.learning_rate
    optim = tf.keras.optimizers.Adam(LR)
    gamma = args.gamma_total_loss
    total_loss = dice_loss + (gamma * binary_focal_loss)
    metrics = [iou_score, f1_score, precision, recall]

    model = Attention_UNet(BACKBONE, classes=1, 
                    input_shape=(32, 128, 128, channels), 
                    encoder_weights='imagenet',
                    activation=activation,
                    dropout=args.dropout_rate
                    )
    model.compile(optimizer = optim, loss=total_loss, metrics=metrics)

    def lr_exp_decay(epoch, lr):
        k = 0.01
        return LR * math.exp(-k*epoch)
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
    checkpoint_filepath = args.log_dir
    filename = 'my_model.epoch{epoch:02d}-loss{val_iou_score:.2f}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath,filename),
        save_weights_only=True,
        monitor='val_iou_score',
        mode='max',
        save_best_only=True,
        verbose = 1)

    X_train, X_test, y_train, y_test = dataset_loader(args.img_dir, args.mask_dir)
    model.fit(X_train, 
            y_train,
            batch_size=args.batch_size, 
            epochs=args.epochs,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1), early_stopping, model_checkpoint_callback])



if __name__ == "__main__":
    main()