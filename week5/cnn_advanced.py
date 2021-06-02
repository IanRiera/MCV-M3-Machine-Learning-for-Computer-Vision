from keras.optimizers import Adam, SGD, RMSprop
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, Activation
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import initializers
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import confusion_matrix
import os
import getpass

from utils import *

from keras.optimizers import SGD

from keras import activations

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle as cPickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.layers import BatchNormalization
import pickle

results_dir = '../results/advanced/'
log_dir = '../log/advanced/'
weights_dir = '../weights/advanced/'
model_dir = '../models/advanced/'

train_data_dir = '/home/mcv/datasets/MIT_split/train'
val_data_dir = '/home/mcv/datasets/MIT_split/test'
test_data_dir = '/home/mcv/datasets/MIT_split/test'
img_size= 64
batch_size = 16
number_of_epoch = 100
validation_samples = 807

# weight initialization
initializer = initializers.glorot_normal()
# initializer = initializers.he_normal()

# create the base pre-trained model
#Build the Multi Layer Perceptron model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=(img_size, img_size, 3), kernel_initializer=initializer,name='conv_1'))
model.add(Activation(activations.relu))
model.add(BatchNormalization(name='batch_1'))

model.add(MaxPooling2D(pool_size=(2, 2),name='max_pool1'))

model.add(Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=(img_size, img_size, 3), kernel_initializer=initializer,name='conv_2'))
model.add(Activation(activations.relu))
model.add(BatchNormalization(name='batch_2'))

model.add(Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=(img_size, img_size, 3), kernel_initializer=initializer,name='conv_3'))
model.add(Activation(activations.relu))
model.add(BatchNormalization(name='batch_3'))

model.add(MaxPooling2D(pool_size=(2, 2),name='max_pool2'))

model.add(Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=(img_size, img_size, 3), kernel_initializer=initializer,name='conv_4'))
model.add(Activation(activations.relu))
model.add(BatchNormalization(name='batch_4'))

model.add(Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=(img_size, img_size, 3), kernel_initializer=initializer,name='conv_5'))
model.add(Activation(activations.relu))
model.add(BatchNormalization(name='batch_5'))

model.add(MaxPooling2D(pool_size=(2, 2),name='max_pool3'))

model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=(img_size, img_size, 3), kernel_initializer=initializer,name='conv_6'))
model.add(Activation(activations.relu))
model.add(BatchNormalization(name='batch_6'))

model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=(img_size, img_size, 3), kernel_initializer=initializer,name='conv_7'))
model.add(Activation(activations.relu))
model.add(BatchNormalization(name='batch_7'))

model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=(img_size, img_size, 3), kernel_initializer=initializer,name='conv_8'))
model.add(Activation(activations.relu))
model.add(BatchNormalization(name='batch_8'))

model.add(MaxPooling2D(pool_size=(2, 2),name='max_pool4'))

model.add(Dropout(0.5))
model.add(Conv2D(8, kernel_size=(1, 1), strides=1, activation='relu', padding='valid', kernel_initializer=initializer,name='conv_9'))
model.add(Activation(activations.relu))
model.add(BatchNormalization(name='batch_9'))

# model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2),padding='same',name='pool'))
# model.add(GlobalMaxPooling2D())
model.add(GlobalAveragePooling2D())
# model.add(Flatten())
model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file=results_dir +'model_baseline.png', show_shapes=True, show_layer_names=True)

print('Done!\n')

# sys.exit()

print(model.summary())


#model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9), metrics=['accuracy'])


# preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             preprocessing_function=preprocess_input,
                             rotation_range=0.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_size, img_size),
                                              batch_size=batch_size,
                                              class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
                                             target_size=(img_size, img_size),
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=False)

validation_generator = datagen.flow_from_directory(val_data_dir,
                                                   target_size=(img_size, img_size),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

tbCallBack = TensorBoard(log_dir=log_dir + 'tboriginal',
                         histogram_freq=10,
                         write_graph=True,
                         profile_batch=0)

save_callback = ModelCheckpoint(filepath=weights_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')

earlyCallBack = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=4,
                              verbose=1,
                              mode='min',
                              baseline=None,
                              restore_best_weights=True)

history = model.fit_generator(train_generator,
                              steps_per_epoch=(int(400 // batch_size) + 1),
                              nb_epoch=number_of_epoch,
                              validation_data=validation_generator,
                              validation_steps=(int(validation_samples // batch_size) + 1),
                              callbacks=[tbCallBack, save_callback])

print('Saving the weights into '+weights_dir+' \n')
model.save_weights(weights_dir+"model_weights")  # always save your weights after training or during training


_, validation_accuracy = model.evaluate_generator(test_generator, val_samples=validation_samples)
print("validation accuracy = {}".format(validation_accuracy))


# save accuracy and loss plot curves
save_accuracy(history, results_dir, baseline=0.78, legend_name='Baseline', xmax=number_of_epoch)
save_loss(history, results_dir, baseline=1.2 ,legend_name='Baseline', xmax=number_of_epoch)

num_parameters = model.count_params()
ratio = validation_accuracy / (num_parameters / 100000)
print("This model advanced has {} parameters and a ratio of {}".format(num_parameters, ratio))
