from keras.optimizers import Adam, SGD, RMSprop
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
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
from PIL import Image
from pathlib import Path

results_dir = '../results/baseline_tuning/std_feat_'
log_dir = '../log/baseline_tuning/'
weights_dir = '../weights/baseline_tuning/std_feat_'
model_dir = '../models/baseline_tuning/std_feat_'

train_data_dir = '/home/mcv/datasets/MIT_split/train'
val_data_dir = '/home/mcv/datasets/MIT_split/test'
test_data_dir = '/home/mcv/datasets/MIT_split/test'
img_size= 32
batch_size = 16
number_of_epoch = 100
validation_samples = 807

# weight initialization
initializer = initializers.glorot_normal()


# create the base pre-trained model
#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',input_shape=(img_size, img_size, 3), kernel_initializer=initializer,name='first'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name='second'))
model.add(Conv2D(32,  kernel_size=(3, 3), activation='relu', kernel_initializer=initializer,name='third'))
model.add(MaxPooling2D(pool_size=(2, 2),name='fourth'))
model.add(Flatten())
model.add(Dense(units=8, activation='softmax'))




model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file=results_dir +'model_baseline.png', show_shapes=True, show_layer_names=True)

print('Done!\n')


print(model.summary())



datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=True,
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
#load train data

def read_pil_image(img_path, height, width):
        with open(img_path, 'rb') as f:
            return np.array(Image.open(f).convert('RGB').resize((width, height)))

def load_all_images(dataset_path, height, width, img_ext='jpg'):
    return np.array([read_pil_image(str(p), height, width) for p in 
                                    Path(dataset_path).rglob("*."+img_ext)]) 
    
                             
datagen.fit(load_all_images(train_data_dir, img_size, img_size))
#
# Obtain mean and std
train_mean = datagen.mean
train_std = datagen.std

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
print("This model std_feat has {} parameters and a ratio of {}".format(num_parameters, ratio))
print("This model std_ has {} parameters and a mean of {}".format(train_std, train_mean))