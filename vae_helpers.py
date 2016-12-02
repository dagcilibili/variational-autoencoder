from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10
import collections

import numpy as np

import time

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

class DataSet(object):
    def __init__(self,
               images,
               labels,
               one_hot=False,
               dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)

        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            if images.shape[0] >0 and np.max(images) > 1:
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        # increase the index in epoch by the batch size
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def make_cifar10_dataset(cifar_dir,n_validation = 0,vectorize=False):
    NUM_CLASSES = 10
    NUM_TRAIN   = 50000
    NUM_TEST    = 10000

    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # reshape to vectors
    if vectorize:
        X_train = np.reshape(X_train,(X_train.shape[0],-1))
        X_test  = np.reshape(X_test,(X_test.shape[0],-1))

    # make one-hot coding
    y_train_temp = np.zeros((NUM_TRAIN,NUM_CLASSES))
    for i in range(NUM_TRAIN):
        y_train_temp[i,y_train[i]] = 1
    y_train = y_train_temp

    y_test_temp = np.zeros((NUM_TEST,NUM_CLASSES))
    for i in range(NUM_TEST):
        y_test_temp[i,y_test[i]] = 1
    y_test = y_test_temp

    # make validation set
    X_valid = X_train[:n_validation]
    X_train = X_train[n_validation:]

    y_valid = y_train[:n_validation]
    y_train = y_train[n_validation:]

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)

def read_cifar10_dataset(cifar_dir,n_validation = 0,vectorize=False):
    X_train, y_train, X_valid, y_valid, X_test, y_test = make_cifar10_dataset(cifar_dir,n_validation,vectorize)
    train_    = DataSet(X_train, y_train)
    test_     = DataSet(X_test, y_test)
    validate_ = DataSet(X_valid, y_valid)

    return Datasets(train=train_,validation=validate_,test=test_)

def reduce_training_set(datasets, num_train = 10):
    X_train = datasets.train.images
    y_train = datasets.train.labels

    X_train_new = X_train[0:num_train]
    y_train_new = y_train[0:num_train]

    train_ = DataSet(X_train_new, y_train_new)
    validate_ = datasets.validation
    test_ = datasets.test

    return Datasets(train=train_,validation=validate_,test=test_)

def print_data_shapes(datasets):
    print('Training data shape: {}'.format(datasets.train.images.shape))
    print('Training labels shape: {}'.format(datasets.train.labels.shape))
    print('Validation data shape: {}'.format(datasets.validation.images.shape))
    print('Validation labels shape: {}'.format(datasets.validation.labels.shape))
    print('Test data shape: {}'.format(datasets.test.images.shape))
    print('Test labels shape: {}'.format(datasets.test.labels.shape))

# to be used in ipython notebook
def visualize_dataset(dataset,height=0,width=0,channels=0):
    images = dataset.images
    labels = dataset.labels
    num_classes = labels.shape[1]
    samples_per_class = 7

    for cls in range(num_classes):
        idxs = np.flatnonzero(labels[:,cls] == 1)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + cls + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            if channels == 1:
                plt.imshow(images[idx].reshape((height,width)))
            elif channels > 1:
                plt.imshow(images[idx].reshape((height,width,channels)))
            else:
                plt.imshow(images[idx])
            plt.axis('off')
            if i == 0:
                plt.title('C{}'.format(cls))
    plt.show()

def get_time_stamp():
    date_string = time.strftime("%Y_%m_%d_%H_%M_%S")
    return date_string