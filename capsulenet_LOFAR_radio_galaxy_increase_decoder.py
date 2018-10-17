"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.

Usage:
       ... ...
python capsulenet_LOFAR_radio_galaxy_increase_filtersize.py --save_dir LOFAR_default_caps_13_10_18_inc_filtersize --n_test 600 --aug 'F' --path_to_npy_data "/path/to/npy/data/train_test_X_6_class_orig_aug_7_9_18_cleaner.npy" --path_to_labels "/path/to/labels/train_test_Y_6_class_orig_aug_7_9_18_cleaner.npy"
           
Author: Vesna Lukic, E-mail: `vlukic973@gmail.com`, originally based on code by Xifeng Guo Github: `https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulenet.py`

"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from matplotlib import cm
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
import os
import time
import datetime

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(1024, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(2048, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 5.],
                  metrics={'capsnet': 'accuracy'})

    
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    
    time_date=datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    filename=time_date+'_capsule_network'+'_batch_size='+str(args.batch_size)+'_epochs='+str(args.epochs)+'_ntr='+str(x_train.shape[0])+'_nch='+str(x_train.shape[3])
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    img = combine_images(np.concatenate([np.abs(x_test[:50]),np.abs(x_recon[:50])]))
    image = cm.hot(img) * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon4.png")
    print('Reconstructed images are saved to %s/real_and_recon4.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)

    predicted_classes=y_pred
    predicted_classes=np.asarray(predicted_classes)	
    b = np.zeros_like(predicted_classes)
    b[np.arange(len(predicted_classes)), predicted_classes.argmax(1)] = 1
    predicted_classes=np.argmax(np.round(b),axis=1)
    b = np.zeros_like(y_test)
    b[np.arange(len(y_test)), y_test.argmax(1)] = 1
    y_test=np.argmax(np.round(b),axis=1)
    class_names=['Unresolved','FRI','FRII']
    from sklearn.metrics import classification_report
    print(classification_report(y_test, predicted_classes, target_names=class_names,digits=4))
    os.chdir(args.save_dir)
    filename1=filename+'_classification_report.txt'
    f = open(filename1,'w')
    f.write(classification_report(y_test, predicted_classes, target_names=class_names,digits=4))

    f = open(filename1,'w')
    f.write(classification_report(y_test, predicted_classes, target_names=class_names,digits=4))

    ############# Architecture

    with open(filename + '_architecture.txt','w') as fh:
	model.summary(print_fn=lambda x: fh.write(x + '\n'))

    with open(filename + '_architecture.txt','w') as fh:
	model.summary(print_fn=lambda x: fh.write(x + '\n'))


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 3, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = cm.hot(img)*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_LOFAR_radio_galaxy(n_test, aug, data, labels):

    import pandas as pd
    import numpy as np
    import random

    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)

    orig_aug_x_class=pd.read_csv("orig_aug_solutions_15_10_18.csv")

    orig_x_class=orig_aug_x_class[orig_aug_x_class['source'].str.endswith('_zmm.npy')]

    orig_x_class=pd.read_csv("shuffle_data_1.csv")

    orig_x_class.set_index('Unnamed: 0', inplace=True)

    if (aug=="T"):

        aug_x_class=orig_aug_x_class[~orig_aug_x_class['source'].str.endswith('_zmm.npy')]

    else:

        aug_x_class=orig_aug_x_class[orig_aug_x_class['source'].str.endswith('no_aug.npy')]

    valid_test_idx=list(orig_x_class.index)
    valid_test_idx1=valid_test_idx[0:n_test]

    remove_valid_test=orig_aug_x_class.iloc[valid_test_idx1]

    remove_valid_test_list=[]

    for i in range(0,len(remove_valid_test)):
	remove_valid_test_list.append(remove_valid_test['source'].tolist()[i][0:18]+'_zmm.npy')

    train_idx1=np.asarray(aug_x_class.index)

    train_idx2=np.asarray(valid_test_idx[n_test:len(valid_test_idx)])
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

    list(set(train_idx) & set(valid_test_idx1))

    orig_aug=np.load(data)
    labels= np.load(labels)

    orig_aug1=orig_aug[:,50:150,50:150,0:3]

    train_X=orig_aug1[train_idx]
    train_Y=labels[train_idx]
    test_X=orig_aug1[valid_test_idx1]
    test_Y=labels[valid_test_idx1]

    train_X[np.isnan(train_X)] = 0
    test_X[np.isnan(test_X)] = 0

    train_X.shape,train_Y.shape,test_X.shape,test_Y.shape

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

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on LOFAR radio galaxies.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=2, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--n_test', default=600, help="The number of samples for validation/testing",type=int)
    parser.add_argument('--aug', default="F", help="use augmented data in addition to original images (T)")
    parser.add_argument('--path_to_npy_data',help="specify path of numpy array data")
    parser.add_argument('--path_to_labels',help="specify path of numpy array labels")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_LOFAR_radio_galaxy(n_test=args.n_test, aug=args.aug, data=args.path_to_npy_data, labels=args.path_to_labels)

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)
