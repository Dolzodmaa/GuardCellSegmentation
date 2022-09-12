"""
  Author: Dolzodmaa Davaasuren
  Attention gated, patch-wise 3D segmentation model
  
  Base structure was on U-Net model

  # Reference paper

- [U-Net: Convolutional Networks for Biomedical Image Segmentation]
  (https://arxiv.org/abs/1505.04597) (MICCAI 2015)

  # Reference implementation of 3D U-Net
  [Segmentation Models 3D]
- https://github.com/ZFTurbo/segmentation_models_3D

"""

from common import Conv3dBn
from tensorflow.keras import backend as K
from keras import backend as K
from backbone import Backbone
import tensorflow as tf
from tensorflow import keras

backend=keras.backend
layers=keras.layers
models=keras.models
utils=keras.utils

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': utils,
    }


def freeze_model(model, **kwargs):
 
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv3dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def repeat_elem(tensor, rep):

    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4),
                          arguments={'repnum': rep})(tensor)


def gating_signal(input, out_size, batch_norm=False):

    x = layers.Conv3D(out_size, (1, 1, 1), strides=1, padding='same', kernel_initializer="he_normal")(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    #print('shape_x', shape_x)
    shape_g = K.int_shape(gating)

    theta_x = layers.Conv3D(inter_shape, kernel_size = 1, strides = 2, padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)
    #print('theta_x', shape_theta_x)

    phi_g = layers.Conv3D(inter_shape, (1, 1, 1), strides = 1, padding='same')(gating)

    #print('phi_g', K.int_shape(phi_g))
    concat_xg = layers.add([phi_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv3D(inter_shape, (1, 1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    #print('shape_sigmoid', shape_sigmoid)
    #sigmoid_x = layers.Lambda(lambda x: x[0, :,: ,:, :])(sigmoid_xg)
    upsample_psi = layers.UpSampling3D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]), data_format="channels_last")(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[4]//shape_sigmoid[4] )
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv3D(shape_x[4], (1, 1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def DecoderAttnTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 4 if keras.backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):
        
        x = layers.Conv3DTranspose(
            filters,
            kernel_size=(4, 4, 4),
            strides=(2, 2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            g = gating_signal(input_tensor, filters)
            attn = attention_block(skip, g, filters)
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, attn])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer

def DecoderAttnUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 4 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):

        x = layers.UpSampling3D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            g = gating_signal(input_tensor, filters)
            attn = attention_block(skip, g, filters)
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, attn])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        dropout=None,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])
    

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    if dropout:
        x = layers.SpatialDropout3D(dropout, name='pyramid_dropout')(x)

    # model head (define number of output classes)
    x = layers.Conv3D(
        filters=classes,
        kernel_size=(3, 3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


def Model(backbone_name='densenet',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='transpose',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        dropout=None,
        **kwargs):
    '''
    Patch-wise Attention 3DUNet, 
    
    '''
    
    if decoder_block_type == 'upsampling':
        decoder_block = DecoderAttnUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderAttnTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbone.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )
    print(backbone)

    if encoder_features == 'default':
        encoder_features = Backbone.get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
        dropout=dropout,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(Backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
