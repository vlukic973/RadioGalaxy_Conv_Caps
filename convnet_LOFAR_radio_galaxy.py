"""
Keras implementation of various ConvNet architectures

Usage:
       ... ...
python convnet_LOFAR_radio_galaxy.py --n_test 600 --aug 'F' --save_dir /path/to/save_dir/ --epochs 3 --use_model 'simple' --path_to_npy_data "/path/to/npy/data/train_test_X_6_class_orig_aug_7_9_18_cleaner.npy" --path_to_labels "/path/to/labels/train_test_Y_6_class_orig_aug_7_9_18_cleaner.npy"
           
Author: Vesna Lukic, E-mail: `vlukic973@gmail.com`

"""

from __future__ import division

if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Convolutional Network on LOFAR radio galaxies.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.0, type=float,
                        help="The value multiplied by lr at each epoch")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result/')
    parser.add_argument('--n_test', default=600, help="The number of samples for validation/testing",type=int)
    parser.add_argument('--aug', default="F", help="use augmented data in addition to original images (T)")
    parser.add_argument('--use_model', default="4 conv", help="which ConvNet architecture to use")
    parser.add_argument('--load_weights', default="F", help="load pretrained weights?")
    parser.add_argument('--path_to_weights',help="specify path of pretrained weights")
    parser.add_argument('--path_to_npy_data',help="specify path of numpy array data")
    parser.add_argument('--path_to_labels',help="specify path of numpy array labels")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

orig_aug_x_class=pd.read_csv("/lofar5/stvf319/LOFAR/Judith_cutouts/Code/github/upload_github/orig_aug_solutions_15_10_18.csv")

orig_x_class=pd.read_csv("/lofar5/stvf319/LOFAR/Judith_cutouts/Code/github/shuffle_data_1.csv")

orig_x_class.set_index('Unnamed: 0', inplace=True)

if (args.aug=="T"):

    aug_x_class=orig_aug_x_class[~orig_aug_x_class['source'].str.endswith('_zmm.npy')]

else:

    aug_x_class=orig_aug_x_class[orig_aug_x_class['source'].str.endswith('no_aug.npy')]

valid_test_idx=list(orig_x_class.index)

valid_test_idx1=valid_test_idx[0:args.n_test]

remove_valid_test=orig_aug_x_class.iloc[valid_test_idx1]

remove_valid_test_list=[]

for i in range(0,len(remove_valid_test)):
    remove_valid_test_list.append(remove_valid_test['source'].tolist()[i][0:18]+'_zmm.npy')

train_idx1=np.asarray(aug_x_class.index)

train_idx2=np.asarray(valid_test_idx[args.n_test:len(valid_test_idx)])
train_idx=np.concatenate((train_idx1, train_idx2), axis=0)

remove_from_train=remove_valid_test_list

train_set=orig_aug_x_class.iloc[train_idx]

train_cat_list=[]

for i in range(0,len(train_set)):
    train_cat_list.append(train_set['source'].tolist()[i][0:18]+'_zmm.npy')

train_cat_list=pd.DataFrame(train_cat_list)

train_set = train_set.assign(cat_list=train_cat_list.values)

train_set=train_set[~train_set['cat_list'].isin(remove_from_train)]

train_idx=train_set.index

train_idx=np.asarray(train_idx)
np.random.shuffle(train_idx)

train_idx=pd.Series(train_idx)

print(list(set(train_idx) & set(valid_test_idx1)))

orig_aug=np.load(args.path_to_npy_data)
labels= np.load(args.path_to_labels)

orig_aug1=orig_aug[:,50:150,50:150,0:3]

train_X=orig_aug1[train_idx]
train_Y=labels[train_idx]
test_X=orig_aug1[valid_test_idx1]
test_Y=labels[valid_test_idx1]

train_X[np.isnan(train_X)] = 0
test_X[np.isnan(test_X)] = 0

print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

classes = np.unique(train_Y)
nClasses = len(classes)

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
     
x_train=train_X
y_train=train_Y_one_hot
x_test=test_X
y_test=test_Y_one_hot

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import regularizers
from keras.layers import Activation
from keras.applications.inception_resnet_v2 import InceptionResNetV2

###### 8 stacked conv layers

if (args.use_model=="8 conv"):

	radio_galaxy_model = Sequential()
	radio_galaxy_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=(train_X.shape[1],train_X.shape[2],train_X.shape[3])))
	radio_galaxy_model.add(Conv2D(32, (3, 3),activation='relu',padding='same'))
	radio_galaxy_model.add(MaxPooling2D((2, 2),padding='same'))
	radio_galaxy_model.add(Dropout(0.25))
	radio_galaxy_model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
	radio_galaxy_model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
	radio_galaxy_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	radio_galaxy_model.add(Dropout(0.25))
	radio_galaxy_model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))  
	radio_galaxy_model.add(Conv2D(128, (3, 3), activation='relu',padding='same')) 
	radio_galaxy_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	radio_galaxy_model.add(Dropout(0.25))
	radio_galaxy_model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))  
	radio_galaxy_model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))  
	radio_galaxy_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	radio_galaxy_model.add(Dropout(0.25))
	radio_galaxy_model.add(Flatten())
	radio_galaxy_model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01)))   
	radio_galaxy_model.add(Dropout(0.5))
	radio_galaxy_model.add(Dense(nClasses, activation='softmax'))

	print(radio_galaxy_model.summary())

if (args.use_model=="4 conv"):

	radio_galaxy_model = Sequential()
	radio_galaxy_model.add(Conv2D(16, (5, 5), activation='relu',padding='same',input_shape=(train_X.shape[1],train_X.shape[2],train_X.shape[3])))
	radio_galaxy_model.add(Conv2D(16, (5, 5), activation='relu',padding='same'))
	radio_galaxy_model.add(MaxPooling2D((2, 2),padding='same'))
	radio_galaxy_model.add(Dropout(0.25))
	radio_galaxy_model.add(Conv2D(16, (5, 5), activation='relu',padding='same'))
	radio_galaxy_model.add(Conv2D(16, (5, 5), activation='relu',padding='same'))
	radio_galaxy_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	radio_galaxy_model.add(Dropout(0.25))
	radio_galaxy_model.add(Flatten())
	radio_galaxy_model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01)))   
	radio_galaxy_model.add(Dropout(0.5))
	radio_galaxy_model.add(Dense(nClasses, activation='softmax'))

	print(radio_galaxy_model.summary())

if (args.use_model=="simple"):

	radio_galaxy_model = Sequential()

        radio_galaxy_model.add(Conv2D(16,(4,4), input_shape=(100,100,3), padding='same'))
        radio_galaxy_model.add(Conv2D(16,(4,4), padding='same'))
        radio_galaxy_model.add(Conv2D(16,(4,4), padding='same'))
        radio_galaxy_model.add(MaxPooling2D((4,4), padding='same'))
        radio_galaxy_model.add(Dropout(0.2))
        radio_galaxy_model.add(Conv2D(16,(4,4), padding='same'))
        radio_galaxy_model.add(Conv2D(16,(4,4), padding='same'))
        radio_galaxy_model.add(MaxPooling2D((4,4), padding='same'))
        radio_galaxy_model.add(Dropout(0.2))
        radio_galaxy_model.add(Flatten())
        radio_galaxy_model.add(Dense(3, activation='softmax'))

if (args.use_model=="transfer"):

	npad=((0, 0),(20, 20),(20, 20),(0,0))

	x_train = np.pad(x_train, pad_width=npad, mode='constant', constant_values=0)
	x_test = np.pad(x_test, pad_width=npad, mode='constant', constant_values=0)

	model_inc = InceptionResNetV2(include_top=False, weights='imagenet', pooling='max',input_shape=(140,140,3))
	x = model_inc.output
	x = Dropout(0.2)(x)
	x = Dense(20, activation="relu")(x)
	pred = Dense(3, activation="softmax")(x)
	model = Model(input = model_inc.input, output = pred)
	radio_galaxy_model=model

	print(radio_galaxy_model.summary())

radio_galaxy_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=args.lr, decay=args.lr_decay),metrics=['accuracy'])

from keras.models import load_model

from keras.callbacks import ModelCheckpoint

import time
import datetime

time_date=datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

if (args.load_weights=='T'):

	radio_galaxy_model.load_weights(args.path_to_weights)

elif (args.load_weights=='F'):

	weight_save_callback = ModelCheckpoint(args.save_dir+'weights.{epoch:02d}_acc-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='auto')

	radio_galaxy_train2 = radio_galaxy_model.fit(x_train, y_train, batch_size=args.batch_size,epochs=args.epochs,verbose=1,validation_data=(x_test, y_test),callbacks=[weight_save_callback])

	os.chdir(args.save_dir)
	filename=time_date+'_double_stacked'+'_batch_size='+str(args.batch_size)+'_epochs='+str(args.epochs)+'_ntr='+str(x_train.shape[0])+'_nch='+str(x_train.shape[3])

### save loss and accuracy curves

	accuracy = radio_galaxy_train2.history['acc']
	val_accuracy = radio_galaxy_train2.history['val_acc']
	loss = radio_galaxy_train2.history['loss']
	val_loss = radio_galaxy_train2.history['val_loss']
	epochs = range(len(accuracy))

	np.save('accuracy.npy',accuracy)
	np.save('val_accuracy.npy',val_accuracy)
	np.save('loss.npy',loss)
	np.save('val_loss.npy',val_loss)
	np.save('epochs.npy',epochs)

############# Training and validation accuracy and loss

	plt.figure()
	plt.plot(epochs, accuracy, 'b', label='Training accuracy')
	plt.plot(epochs, val_accuracy, 'bo', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.savefig(args.save_dir+filename+'_accuracy.png')
	plt.clf()

	plt.figure()
	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'bo', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig(args.save_dir+filename+'_loss.png')
	plt.clf()

######## Test score

filename_test=filename+'_test_score.txt'
f = open(filename_test,'w')
score=radio_galaxy_model.evaluate(x_test, y_test, batch_size=args.batch_size)
f.write(str(score))

############# Classification report

predicted_classes = radio_galaxy_model.predict(x_test)

b = np.zeros_like(predicted_classes)
b[np.arange(len(predicted_classes)), predicted_classes.argmax(1)] = 1

predicted_classes=np.argmax(np.round(b),axis=1)

predicted_classes.shape, test_Y.shape

class_names=['Unresolved','FRI','FRII']

from sklearn.metrics import classification_report
print(classification_report(test_Y, predicted_classes, target_names=class_names,digits=4))

filename1=filename+'_classification_report.txt'
f = open(filename1,'w')
f.write(classification_report(test_Y, predicted_classes, target_names=class_names,digits=4))

filename1=filename+'_classification_report.txt'
f = open(filename1,'w')
f.write(classification_report(test_Y, predicted_classes, target_names=class_names,digits=4))


############# Architecture

with open(filename + '_architecture.txt','w') as fh:
# Pass the file handle in as a lambda function to make it callable
	radio_galaxy_model.summary(print_fn=lambda x: fh.write(x + '\n'))

with open(filename + '_architecture.txt','w') as fh:
# Pass the file handle in as a lambda function to make it callable
	radio_galaxy_model.summary(print_fn=lambda x: fh.write(x + '\n'))

############# ROC curve

predicted_classes = radio_galaxy_model.predict(x_test)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(nClasses):
	fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predicted_classes[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])

class_names=['Unresolved','FRI','FRII']

plt.figure(1)
colors = ['red','orange','green']
for i, color in zip(range(nClasses), colors):
	plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='{0} (area = {1:0.2f})'''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig(args.save_dir+filename+'_roc_curve.png')
plt.clf()
