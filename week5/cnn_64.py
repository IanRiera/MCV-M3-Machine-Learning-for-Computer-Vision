from keras.optimizers import Adam, SGD, RMSprop
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout,BatchNormalization
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

import pickle

results_dir = '../results/advanced/64_'
log_dir = '../log/advanced/'
weights_dir = '../weights/advanced/64_'
model_dir = '../models/advanced/64_'

train_data_dir = '/home/mcv/datasets/MIT_split/train'
val_data_dir = '/home/mcv/datasets/MIT_split/test'
test_data_dir = '/home/mcv/datasets/MIT_split/test'
img_size= 64
batch_size = 16
number_of_epoch = 100
validation_samples = 807

# weight initialization
initializer = initializers.glorot_normal()

# create the base pre-trained model
#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',input_shape=(img_size, img_size, 3),padding='same', kernel_initializer=initializer,name='conv1'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name='pool1'))
model.add(Conv2D(32,  kernel_size=(3, 3), activation='relu',padding="same", kernel_initializer=initializer,name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='pool2'))
model.add(Conv2D(64,  kernel_size=(3, 3), activation='relu',padding="same", kernel_initializer=initializer,name='conv3'))
model.add(MaxPooling2D(pool_size=(2, 2),name='pool3'))
model.add(Dropout(0.5))
model.add(Conv2D(64,  kernel_size=(3, 3), activation='relu',padding="same", kernel_initializer=initializer,name='conv4'))
model.add(BatchNormalization())

# model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2),padding='same',name='pool'))
# model.add(GlobalMaxPooling2D())
model.add(GlobalAveragePooling2D())
#model.add(Flatten())
model.add(Dense(units=8, activation='softmax'))


learning_rate = 1e-4
optimizer_init = Adam(learning_rate=learning_rate)


model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_init,
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file=results_dir +'model_baseline.png', show_shapes=True, show_layer_names=True)

print('Done!\n')


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

model.save(model_dir+'model')
_, validation_accuracy = model.evaluate_generator(test_generator, val_samples=validation_samples)
print("validation accuracy = {}".format(validation_accuracy))


# save accuracy and loss plot curves
save_accuracy(history, results_dir, baseline=0.78, legend_name='Baseline', xmax=number_of_epoch)
save_loss(history, results_dir, baseline=1.2 ,legend_name='Baseline', xmax=number_of_epoch)

num_parameters = model.count_params()
ratio = validation_accuracy / (num_parameters / 100000)
print("This model 64 has {} parameters and a ratio of {}".format(num_parameters, ratio))