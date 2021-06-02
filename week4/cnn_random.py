from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import confusion_matrix

from tensorflow.keras.activations import *

import pickle

from kerastuner.tuners import RandomSearch

results_dir = '../results/random/'
log_dir = '../log/random/'
weights_dir = '../weights/random/'

train_data_dir='../mini_dataset/train'
val_data_dir='../mini_dataset/test'
test_data_dir='../mini_dataset/test'
img_width=299 # original: 224 -> we have to change it to 299 to fit the inception_v3 model input shape
img_height=299
batch_size=32
number_of_epoch=30
validation_samples=807

#preprocessing_function=preprocess_input,
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
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

tbCallBack = TensorBoard(log_dir=log_dir+'tboriginal',
                         histogram_freq=10,
                         write_graph=True,
                         profile_batch=0)

def get_optimizer(optimizer, learning_rate, momentum):
    if optimizer == 'SGD':
        optimizer_init = SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'RMSprop':
        optimizer_init = RMSprop(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'Adam':
        optimizer_init = Adam(learning_rate=learning_rate)
    elif optimizer == 'Adadelta':
        optimizer_init = Adadelta(learning_rate=learning_rate)
    elif optimizer == 'Adagrad':
        optimizer_init = Adagrad(learning_rate=learning_rate)

    return optimizer_init

activ_func_dict = {
    'relu': relu,
    'tanh': tanh,
    'elu': elu
}

def create_model(hp):

    base_model = InceptionV3(weights='imagenet')

    y = base_model.get_layer('mixed5').output
    y = GlobalAveragePooling2D()(y)
    y = Dense(8, activation='softmax', name='predictions')(y)

    model = Model(inputs=base_model.input, outputs=y)

    optimizer_init = get_optimizer(hp.Choice('optimizer', ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad']),
                                   hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3]),
                                   hp.Choice('momentum', [0.6, 0.8, 0.9]))

    activ_func = hp.Choice('dense_activation', ['relu', 'tanh', 'elu'])

    for layer in model.layers:
        if str(layer.__class__.__name__) == "Activation":
            layer.activation = activ_func_dict[activ_func]

    model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer_init,
                      metrics=['accuracy'])

    return model

#####TUNER
tuner = RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3)
    
tuner.search_space_summary()

tuner.search(train_generator, steps_per_epoch=400//batch_size, epochs=number_of_epoch, validation_data=validation_generator)

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

tuner.results_summary()

# Evaluate the best model.
loss, accuracy = best_model.evaluate_generator(test_generator, val_samples=validation_samples)

print(f'loss: {loss}')
print(f'accuracy: {accuracy}')

# VISUALIZATION
test_labels = test_generator.classes
steps = validation_samples // batch_size+1
class_names = ['Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding']

Y_pred = best_model.predict_generator(test_generator, steps)
y_pred = np.argmax(Y_pred, axis=1)

#Confusion Matrix
print('Confusion Matrix')
conf_mat = confusion_matrix(test_labels, y_pred)
conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
save_confusion_matrix(conf_mat,class_names,results_dir+'confusion_matrix.png')

#Classification Report
print('Classification Report')
print(classification_report(test_labels, y_pred, target_names=class_names))

#ROC curve
compute_roc(test_labels, Y_pred, class_names, results_dir+'test_roc.png', 'ROC curve - default InceptionV3')

