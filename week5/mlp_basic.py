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
from scipy.misc import imresize
import pickle as cPickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
WEIGHTS_FNAME = '/home/group04/week3/weights/basic/mlp_basic.h5'
MODEL_FNAME = '/home/group04/week3/models/basic/mlp_basic.h5'
RESULTS = '/home/group04/week3/results/basic/'

if not os.path.exists(DATASET_DIR):
  print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()

print('Building MLP model...\n')

#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
model.add(Dense(units=2048, activation='relu',name='second'))
model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file=RESULTS+'modelMLP.png', show_shapes=True, show_layer_names=True)

print('Done!\n')

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

print('Start training...\n')

# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical',
        shuffle=False)




history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE)

print('Done!\n')
print('Saving the weights into '+WEIGHTS_FNAME+' \n')
#model.save_weights(WEIGHTS_FNAME)  # always save your weights after training or during training
print('Saving the model into '+MODEL_FNAME+' \n')
#model.save(MODEL_FNAME)
print('Done!\n')

  # summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(RESULTS+'accuracy.jpg')
plt.close()
  # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(RESULTS+'loss.jpg')
plt.close()

#to get the output of a given layer
 #crop the model up to a certain layer
model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)

#get the features from images
directory = DATASET_DIR+'/test/coast'
x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0] )))
x = np.expand_dims(imresize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
print('prediction for image ' + os.path.join(directory, os.listdir(directory)[0] ))
features = model_layer.predict(x/255.0)
print(features)
print('Done!')


# VISUALIZATION
validation_samples = 807
test_labels = validation_generator.classes
steps = validation_samples // BATCH_SIZE+1
class_names = ['Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding']

Y_pred = model.predict_generator(validation_generator, steps)
y_pred = np.argmax(Y_pred, axis=1)

#Confusion Matrix
print('Confusion Matrix')
conf_mat = confusion_matrix(test_labels, y_pred)
conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
save_confusion_matrix(conf_mat,class_names,RESULTS+'confusion_matrix.png')

#Classification Report
print('Classification Report')
print(classification_report(test_labels, y_pred, target_names=class_names))

#ROC curve
compute_roc(test_labels, Y_pred,class_names, RESULTS+'test_roc.png', 'ROC curve - MLP')