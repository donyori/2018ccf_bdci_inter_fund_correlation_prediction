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
