# RadioGalaxy_Conv_Caps
Python code for classifying Radio galaxies using ConvNet and CapsNet architectures

Before running any of the python scripts, first download the necessary files:

- train_test_X_6_class_orig_aug_7_9_18_cleaner.npy
- train_test_Y_6_class_orig_aug_7_9_18_cleaner.npy
- orig_aug_solutions_15_10_18.csv
- shuffle_data_1.c
To run the convolutional network achitectures, use the convnet_LOFAR_radio_galaxy.py script. This assumes you have

Example usage:

python convnet_LOFAR_radio_galaxy.py --n_test 600 --aug 'F' --save_dir /path/to/save_dir/ --epochs 3 --use_model 'simple' --path_to_npy_data "/path/to/npy/data/train_test_X_6_class_orig_aug_7_9_18_cleaner.npy" --path_to_labels "/path/to/labels/train_test_Y_6_class_orig_aug_7_9_18_cleaner.npy"

To run the Capsule network architectures, first go to: https://github.com/XifengGuo/CapsNet-Keras

Download the following:

-capsulelayers.py
-utils.py
