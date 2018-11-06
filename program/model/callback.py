import time
from datetime import timedelta

from pytimeparse.timeparse import timeparse
from tensorflow import keras

from .save_and_load import save_model, save_epoch_number


class ModelSaver(keras.callbacks.Callback):

    def __init__(self, model_name, period=1, verbose=0):
        super().__init__()
        self.model_name = model_name
        self.period = period
        self.epochs_since_last_save = 0
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            save_model(model=self.model, model_name=self.model_name, does_overwrite=True, does_include_optimizer=True)
            if self.verbose > 0:
                print('Save the model "%s" successfully.' % self.model_name)


class EpochNumberSaver(keras.callbacks.Callback):

    def __init__(self, model_name, verbose=0):
        super().__init__()
        self.model_name = model_name
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        save_epoch_number(model_name=self.model_name, epoch=epoch)
        if self.verbose > 0:
            print('Save the epoch number successfully.')


class TimeLimiter(keras.callbacks.Callback):

    def __init__(self, limit, verbose=0):
        super().__init__()
        if limit is None:
            raise ValueError('TimeLimiter: limit cannot be None.')
        self.limit = limit
        self.verbose = verbose
        self.train_begin_time = 0.
        self.epoch_begin_time = 0.
        self.epoch_avg_seconds = 0.
        self.epoch_count = 0
        self.stopped_epoch = 0
        self.__parse_limit()

    def on_train_begin(self, logs=None):
        self.train_begin_time = time.time()
        self.epoch_avg_seconds = 0.
        self.epoch_count = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        epoch_seconds = now - self.epoch_begin_time
        self.epoch_avg_seconds = (self.epoch_avg_seconds * self.epoch_count + epoch_seconds) / (self.epoch_count + 1)
        self.epoch_count += 1
        if now - self.train_begin_time + self.epoch_avg_seconds > self.limit:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %d: stop by time limiter.' % (self.stopped_epoch + 1))

    def __parse_limit(self):
        if isinstance(self.limit, str):
            self.limit = timeparse(self.limit)
        elif isinstance(self.limit, timedelta):
            self.limit = self.limit.total_seconds()
        if not isinstance(self.limit, float):
            self.limit = float(self.limit)
