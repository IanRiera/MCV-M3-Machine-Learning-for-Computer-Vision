from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from sklearn.model_selection import cross_validate
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import itertools

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Color:
    GRAY=30
    RED=31
    GREEN=32
    YELLOW=33
    BLUE=34
    MAGENTA=35
    CYAN=36
    WHITE=37
    CRIMSON=38

def colorize(num, string, bold=False, highlight = False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def colorprint(colorcode, text, o=sys.stdout, bold=False):
    o.write(colorize(colorcode, text, bold=bold))

def generate_image_patches_db(in_directory,out_directory,patch_size=64):
  if not os.path.exists(out_directory):
      os.makedirs(out_directory)

  total = 2688
  count = 0
  for split_dir in os.listdir(in_directory):
    if not os.path.exists(os.path.join(out_directory,split_dir)):
      os.makedirs(os.path.join(out_directory,split_dir))

    for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
      if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
        os.makedirs(os.path.join(out_directory,split_dir,class_dir))

      for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
        count += 1
        print('Processed images: '+str(count)+' / '+str(total), end='\r')
        im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
        patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size),max_patches=(int(np.asarray(im).shape[0]/patch_size)**2))#max_patches=1.0
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')


def compute_roc(test_labels, y_score, classes, results_path, title_roc):
    # first we need to binarize the labels
    y_test = LabelBinarizer().fit_transform(test_labels)
    n_classes = y_test.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(8,8))
    lw = 2
    plt.plot(fpr["macro"], tpr["macro"],
            label='average ROC curve (auc = {0:0.3f})'
                  ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    palette = sns.color_palette("hls", 8)
    colors = cycle(palette)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='{0} (auc = {1:0.3f})'
                ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title_roc)
    plt.legend(loc="lower right")

    plt.savefig(results_path)
    plt.close()

def load_dataset(path):
    filenames, labels = [], []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if not os.path.isdir(label_path):
            continue
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)
            if not image_path.endswith('.jpg'):
                continue
            filenames.append(image_path)
            labels.append(label)
    return filenames, labels

def save_confusion_matrix(cm, classes, output_file):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.ylim(len(cm) - 0.65, -0.52)

    plt.savefig(output_file)
    plt.close()

def save_accuracy(history, results_dir, baseline=None, legend_name=None, xmax=20):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    if baseline is not None:
        plt.hlines(baseline, 0, xmax, 'g')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if baseline is not None:
        plt.legend(['train', 'validation', legend_name], loc='lower right')
    else:
        plt.legend(['train', 'validation'], loc="lower right")
    plt.savefig(results_dir+'accuracy.jpg')
    plt.close()

def save_loss(history, results_dir, baseline=None, legend_name=None, xmax=20):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if baseline is not None:
        plt.hlines(baseline, 0, xmax, 'g')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if baseline is not None:
        plt.legend(['train', 'validation', legend_name], loc='upper right')
    else:
        plt.legend(['train', 'validation'], loc="upper right")
    plt.savefig(results_dir+'loss.jpg')
    plt.close()