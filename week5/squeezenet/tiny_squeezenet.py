import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, concatenate, Activation, ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from keras import initializers

def FireModule(s_1x1, e_1x1, e_3x3, name):
    """FireModule
        Fire module for the SqueezeNet model. 
        Implements the expand layer, which has a mix of 1x1 and 3x3 filters, 
        by using two conv layers concatenated in the channel dimension. 
    :param s_1x1: Number of 1x1 filters in the squeeze layer
    :param e_1x1: Number of 1x1 filters in the expand layer
    :param e_3x3: Number of 3x3 filters in the expand layer
    :param name: Name of the fire module
    :return: 
        Returns a callable function
    """
    # Concat on the channel axis. TensorFlow uses (rows, cols, channels), while

    concat_axis = 3
    initializer = initializers.glorot_normal()
    def layer(x):
        squeeze = Convolution2D(s_1x1,  kernel_size=(1, 1), activation='relu', kernel_initializer=initializer, name=name+'/squeeze1x1')(x)
        squeeze = BatchNormalization(name=name+'/squeeze1x1_bn')(squeeze)

        # Needed to merge layers expand_1x1 and expand_3x3.
        expand_1x1 = Convolution2D(e_1x1, kernel_size=(1, 1), activation='relu', kernel_initializer=initializer, name=name+'/expand1x1')(squeeze)

        # Pad the border with zeros. Not needed as padding='same' will do the same.
        # expand_3x3 = ZeroPadding2D(padding=(1, 1), name=name+'_expand_3x3_padded')(squeeze)
        expand_3x3 = Convolution2D(e_3x3,kernel_size=(3, 3), padding='same', activation='relu',  kernel_initializer=initializer, name=name+'/expand3x3')(squeeze)
        # Concat in the channel dim
        expand_merge = concatenate([expand_1x1, expand_3x3],  axis=concat_axis, name=name+'/concat')
        return expand_merge
    return layer
    

def SqueezeNet(nb_classes=8, rows=32, cols=32, channels=3):
    """
        SqueezeNet v1.1 implementation
    :param nb_classes: Number of classes
    :param rows: Amount of rows in the input
    :param cols: Amount of cols in the input
    :param channels: Amount of channels in the input
    :returns: SqueezeNet model
    """
    input_shape = (rows, cols, channels)
    initializer = initializers.glorot_normal()

    input_image = Input(shape=input_shape)
    # conv1 output shape = (113, 113, 64)
    conv1 = Convolution2D(64, kernel_size=(3, 3), activation='relu',  kernel_initializer=initializer, name='conv1')(input_image)
    # maxpool1 output shape = (56, 56, 64)
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(conv1)
    # fire2 output shape = (?, 56, 56, 128)
    fire2 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire2')(maxpool1)
    # fire3 output shape = (?, 56, 56, 128)
    fire3 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire3')(fire2)
    '''
    # maxpool3 output shape = (?, 27, 27, 128)
    maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(fire3)
    # fire4 output shape = (?, 56, 56, 256)
    fire4 = FireModule(s_1x1=32, e_1x1=128, e_3x3=128, name='fire4')(fire3)
    # fire5 output shape = (?, 56, 56, 256)
    fire5 = FireModule(s_1x1=32, e_1x1=128, e_3x3=128, name='fire5')(fire4)
    # maxpool5 output shape = (?, 27, 27, 256)
    maxpool5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(fire5)
    # fire6 output shape = (?, 27, 27, 384)
    fire6 = FireModule(s_1x1=48, e_1x1=192, e_3x3=192, name='fire6')(maxpool5)
    # fire7 output shape = (?, 27, 27, 384)
    fire7 = FireModule(s_1x1=48, e_1x1=192, e_3x3=192, name='fire7')(fire6)
    # fire8 output shape = (?, 27, 27, 512)
    fire8 = FireModule(s_1x1=64, e_1x1=256, e_3x3=256, name='fire8')(fire7)
    # fire9 output shape = (?, 27, 27, 512)
    fire9 = FireModule(s_1x1=64, e_1x1=256, e_3x3=256, name='fire9')(fire8)
    '''
    # Dropout after fire9 module.
    dropout9 = Dropout(0.5, name='dropout9')(fire3)
    # conv10 output shape = (?, 27, 27, nb_classes)
    conv10 = Convolution2D(nb_classes, kernel_size=(1, 1), activation='relu',  kernel_initializer=initializers.he_normal(), name='conv10')(dropout9)
    conv10 = BatchNormalization(name='conv10_bn')(conv10)
    # avgpool10, softmax output shape = (?, nb_classes)
    avgpool10 = GlobalAveragePooling2D(name='pool10')(conv10)
    softmax = Activation('softmax', name='loss')(avgpool10)

    model = Model(inputs=input_image, outputs=[softmax])
    return model
