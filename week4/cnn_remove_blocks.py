from keras.optimizers import Adam, SGD, RMSprop
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import confusion_matrix

import pickle

results_dir = '../results/remove_blocks/'
log_dir = '../log/'
weights_dir = '../weights/remove_blocks/'

train_data_dir='/home/mcv/datasets/MIT_split/train'
val_data_dir='/home/mcv/datasets/MIT_split/test'
test_data_dir='/home/mcv/datasets/MIT_split/test'
img_width=299 # original: 224 -> we have to change it to 299 to fit the inception_v3 model input shape
img_height=299
batch_size=32
number_of_epoch=60
validation_samples=807
    
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet')
# plot_model(base_model, to_file=results_dir+'modelInceptionV3original.png', show_shapes=True, show_layer_names=True)

y = base_model.get_layer('mixed5').output
y = GlobalAveragePooling2D()(y)
y = Dense(8, activation='softmax',name='predictions')(y)

new_model = Model(input=base_model.input, output=y)

print(new_model.summary())
    
new_model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.001, momentum=0.9), metrics=['accuracy'])
for layer in new_model.layers:
    print(layer.name, layer.trainable)

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
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

save_callback = ModelCheckpoint(filepath=weights_dir+'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
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

history=new_model.fit_generator(train_generator,
        steps_per_epoch=(int(400//batch_size)+1),
        epochs=number_of_epoch,
        validation_data=validation_generator,
        validation_steps= (int(validation_samples//batch_size)+1), callbacks=[tbCallBack, save_callback])


result = new_model.evaluate_generator(test_generator, val_samples=validation_samples)
print(result)

# save accuracy and loss plot curves
save_accuracy(history, results_dir)
save_loss(history, results_dir)

# VISUALIZATION
test_labels = test_generator.classes
steps = validation_samples // batch_size+1
class_names = ['Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding']

Y_pred = new_model.predict_generator(test_generator, steps)
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

"""

# TO OBTAIN THE ACTIVATION MAPS FROM THE INNER LAYERS
# forest: sclos13.jpg, mountain: land278.jpg, opencountry: natu921.jpg
img_path = train_data_dir + '/forest/sclos13.jpg'
img = load_img(img_path, target_size=(299,299))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model_features = Model(input=model.input, output=model.get_layer('conv2d_10').output)
features = model_features.predict(x)

with open(results_dir+'forest_features_10.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_20').output)
features = model_features.predict(x)

with open(results_dir+'forest_features_20.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_30').output)
features = model_features.predict(x)

with open(results_dir+'forest_features_30.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_40').output)
features = model_features.predict(x)

with open(results_dir+'forest_features_40.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_50').output)
features = model_features.predict(x)

with open(results_dir+'forest_features_50.pkl', 'wb+') as f:
    pickle.dump(features, f)

img_path = train_data_dir + '/mountain/land278.jpg'
img = load_img(img_path, target_size=(299,299))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model_features = Model(input=model.input, output=model.get_layer('conv2d_10').output)
features = model_features.predict(x)

with open(results_dir+'mountain_features_10.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_20').output)
features = model_features.predict(x)

with open(results_dir+'mountain_features_20.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_30').output)
features = model_features.predict(x)

with open(results_dir+'mountain_features_30.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_40').output)
features = model_features.predict(x)

with open(results_dir+'mountain_features_40.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_50').output)
features = model_features.predict(x)

with open(results_dir+'mountain_features_50.pkl', 'wb+') as f:
    pickle.dump(features, f)

img_path = train_data_dir + '/Opencountry/natu921.jpg'
img = load_img(img_path, target_size=(299,299))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model_features = Model(input=model.input, output=model.get_layer('conv2d_10').output)
features = model_features.predict(x)

with open(results_dir+'opencountry_features_10.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_20').output)
features = model_features.predict(x)

with open(results_dir+'opencountry_features_20.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_30').output)
features = model_features.predict(x)

with open(results_dir+'opencountry_features_30.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_40').output)
features = model_features.predict(x)

with open(results_dir+'opencountry_features_40.pkl', 'wb+') as f:
    pickle.dump(features, f)

model_features = Model(input=model.input, output=model.get_layer('conv2d_50').output)
features = model_features.predict(x)

with open(results_dir+'opencountry_features_50.pkl', 'wb+') as f:
    pickle.dump(features, f)
    
"""
