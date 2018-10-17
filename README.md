# RadioGalaxy_Conv_Caps
Python code for classifying Radio galaxies using ConvNet and CapsNet architectures

Before running any of the python scripts, first make sure keras is installed and download the necessary files:

- train_test_X_6_class_orig_aug_7_9_18_cleaner.npy
- train_test_Y_6_class_orig_aug_7_9_18_cleaner.npy
- orig_aug_solutions_15_10_18.csv
- shuffle_data_1.csv

When running the scripts it is assumed the orig_aug_solutions_15_10_18.csv and shuffle_data_1.csv are in the current directory.

To run the convolutional network achitectures, use the convnet_LOFAR_radio_galaxy.py script. 

ConvNet example usage:

python convnet_LOFAR_radio_galaxy.py --n_test 600 --aug 'F' --save_dir /path/to/save_dir/ --epochs 3 --use_model 'simple' --path_to_npy_data "/path/to/npy/data/train_test_X_6_class_orig_aug_7_9_18_cleaner.npy" --path_to_labels "/path/to/labels/train_test_Y_6_class_orig_aug_7_9_18_cleaner.npy"

The Capsule network architectures are initially based on the capsulenet.py code from https://github.com/XifengGuo/CapsNet-Keras. To run the Capsule network architectures for the radio galaxy data, first go to the link provided and download the following:

- capsulelayers.py
- utils.py

CapsNet example usage:

python capsulenet_LOFAR_radio_galaxy_increase_filtersize.py --save_dir LOFAR_default_caps_13_10_18_inc_filtersize --n_test 600 --aug 'F' --path_to_npy_data "/path/to/npy/data/train_test_X_6_class_orig_aug_7_9_18_cleaner.npy" --path_to_labels "/path/to/labels/train_test_Y_6_class_orig_aug_7_9_18_cleaner.npy"

python capsulenet_LOFAR_radio_galaxy_increase_filtersize.py --save_dir LOFAR_default_caps_13_10_18_inc_filtersize --n_test 600  --aug 'F' -t --digit 2 -w /path/to/LOFAR_default_caps_13_10_18_inc_filtersize_subset/trained_model.h5 --path_to_npy_data "/path/to/npy/data/train_test_X_6_class_orig_aug_7_9_18_cleaner.npy" --path_to_labels "/path/to/labels/train_test_Y_6_class_orig_aug_7_9_18_cleaner.npy"

