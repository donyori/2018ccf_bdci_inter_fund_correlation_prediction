import time
from datetime import timedelta

import numpy as np
from pytimeparse.timeparse import timeparse
from tensorflow import keras
from tensorflow import logging

from .save_and_load import save_model, save_last_epoch_number, save_best_info


class ModelSaver(keras.callbacks.Callback):

    def __init__(self, model_name, period=1, verbose=0):
        super().__init__()
        if not isinstance(period, int):
            raise TypeError('period should be int.')
        if period <= 0:
            raise ValueError('period should be at least 1.')
        self.model_name = model_name
        self.period = period
        self.verbose = verbose
        self.epochs_since_last_save = 0

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            save_model(model_name=self.model_name, model=self.model, epoch=epoch,
                       does_overwrite=True, does_include_optimizer=True)
            if self.verbose > 0:
                print('Epoch %d: save the model "%s" successfully.' % (epoch + 1, self.model_name))


class EpochNumberSaver(keras.callbacks.Callback):

    def __init__(self, model_name, verbose=0):
        super().__init__()
        self.model_name = model_name
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        save_last_epoch_number(model_name=self.model_name, epoch=epoch)
        if self.verbose > 0:
            print('Epoch %d: save the epoch number successfully.' % epoch + 1)


class BestInfoSaver(keras.callbacks.Callback):

    def __init__(self, model_name, monitor='loss', mode='min', baseline=None, verbose=0):
        super().__init__()
        self.model_name = model_name
        self.monitor = monitor
        self.baseline = baseline
        self.verbose = verbose
        if mode is not None:
            mode = mode.lower()
        if mode not in ['min', 'max']:
            logging.warning('BestInfoSaver mode %s is unknown, fallback to min mode.', mode)
            mode = 'min'
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
        self.best = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf
            if self.monitor_op != np.less:
                self.best = -self.best

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('BestInfoSaver conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
            return
        if self.monitor_op(current, self.best):
            self.best = current
            save_best_info(model_name=self.model_name, epoch=epoch, monitor_name=self.monitor, monitor_value=self.best)
            if self.verbose > 0:
                print('Epoch %d: save best info successfully.' % epoch + 1)
        elif self.verbose > 0:
            print('Epoch %d: %s did NOT improve from %s' % (epoch + 1, self.monitor, self.best))


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

        # Allow instances to be re-used
        self.epoch_avg_seconds = 0.
        self.epoch_count = 0
        self.stopped_epoch = 0

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
        now = time.time()
        et = timedelta(seconds=now - self.train_begin_time)
        eta_of_next_epoch = timedelta(seconds=self.epoch_avg_seconds)
        rtl = timedelta(seconds=self.limit - now)
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %d: stop by time limiter. '
                  'Elapsed time: %s, ETA of next epoch: %s, Remaining time limit: %s.' % (
                    self.stopped_epoch + 1, et, eta_of_next_epoch, rtl))

    def __parse_limit(self):
        if isinstance(self.limit, str):
            try:
                self.limit = float(self.limit)
                self.limit *= 60.
            except ValueError:
                self.limit = timeparse(self.limit)
        elif isinstance(self.limit, timedelta):
            self.limit = self.limit.total_seconds()
        if self.limit is None:
            raise ValueError('TimeLimiter: cannot parse limit.')
        if not isinstance(self.limit, float):
            self.limit = float(self.limit)
