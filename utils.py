# DISCLAIMER:
# I do not own and wrote this code!
# Written by @rragundez
# https://github.com/rragundez/PyDataAmsterdam2018/blob/master/utils.py
import os
import random
import time
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

def save_keras_dataset_to_disk(X_train, y_train, X_test, y_test):
    for i, x_img in enumerate(X_train):
        label = y_train[i, 0]
        os.makedirs('data/CIFAR10/train/{}'.format(label), exist_ok=True)
        Image.fromarray(x_img).save("data/CIFAR10/train/{}/{}.jpeg".format(label, i))
    for i, x_img in enumerate(X_test):
        label = y_test[i, 0]
        os.makedirs('data/CIFAR10/test/{}'.format(label), exist_ok=True)
        Image.fromarray(x_img).save("data/CIFAR10/test/{}/{}.jpeg".format(label, i))

class TimeSummary(Callback):
    def on_train_begin(self, logs={}):
        self.epoch_times = []
        self.training_time = time.process_time()

    def on_train_end(self, logs={}):
        self.training_time = time.process_time() - self.training_time

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.process_time()

    def on_epoch_end(self, batch, logs={}):
        self.epoch_times.append(time.process_time() - self.epoch_time_start)
