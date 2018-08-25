# DISCLAIMER:
# I do not own and wrote this code!
# Written by @rragundez
# https://github.com/rragundez/PyDataAmsterdam2018/blob/master/utils.py
import time
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

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
