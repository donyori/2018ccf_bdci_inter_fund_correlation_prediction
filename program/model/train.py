import os
import re

from tensorflow import keras

from constants import PROJECT_HOME
from data.data_generator import SquareExDataGenerator
from data.load_dataset import DATASET_NAME_TRAIN
from .callback import ModelSaver, EpochNumberSaver
from .compile import compile_model
from .constants import MAIN_OUTPUT_NAME
from .metric import custom_metrics
from .save_and_load import load_model, load_epoch_number
from .version.navi import get_rolling_window_size, build_model, get_latest_version_model_name

LOG_DIRECTORY = os.path.join(PROJECT_HOME, 'log')

_remove_pattern = re.compile(r'[\s|_]+', flags=re.UNICODE)

default_batch_size = 200
default_callbacks = ['model_saver', 'epoch_number_saver', 'tensor_board', 'early_stopping']
default_max_queue_size = 10

config = {
    'batch_size': default_batch_size,
    'callbacks': default_callbacks,
    'max_queue_size': default_max_queue_size,
}


def train_model(model_name, model, row_start=None, row_end=None, epoch_number=1, initial_epoch=0,
                worker_number=1, verbose=2):
    if 'batch_size' not in config:
        config['batch_size'] = default_batch_size
    if 'callbacks' not in config:
        config['callbacks'] = default_callbacks
    if 'max_queue_size' not in config:
        config['max_queue_size'] = default_max_queue_size
    callbacks = list() if config['callbacks'] is not None else None
    if callbacks is not None:
        for cb in config['callbacks']:
            if isinstance(cb, keras.callbacks.Callback):
                callbacks.append(cb)
            elif isinstance(cb, str):
                cb_str = cb.lower()
                cb_str = re.sub(pattern=_remove_pattern, repl='', string=cb_str)
                if cb_str == 'earlystopping':
                    callbacks.append(keras.callbacks.EarlyStopping(
                        monitor=MAIN_OUTPUT_NAME+'_loss', min_delta=1e-4, patience=2))
                elif cb_str == 'tensorboard':
                    callbacks.append(keras.callbacks.TensorBoard(
                        log_dir=os.path.join(LOG_DIRECTORY, model_name),
                        batch_size=config['batch_size'],
                        write_graph=True,
                    ))
                elif cb_str == 'modelsaver':
                    callbacks.append(ModelSaver(model_name=model_name, period=1, verbose=1))
                elif cb_str == 'epochnumbersaver':
                    callbacks.append(EpochNumberSaver(model_name=model_name, verbose=1))
                else:
                    raise UnknownCallbackNameException(cb)
            else:
                raise TypeError('Callback must be an instance of keras.callbacks.Callback or a callback name(string).')
    rolling_window_size = get_rolling_window_size(model_name)
    generator = SquareExDataGenerator(
        dataset_name=DATASET_NAME_TRAIN,
        rolling_window_size=rolling_window_size,
        row_start=row_start,
        row_end=row_end,
        max_batch_size=config['batch_size'],
        does_shuffle=True,
    )
    history = model.fit_generator(
        generator=generator,
        epochs=epoch_number + initial_epoch,
        verbose=verbose,
        callbacks=callbacks,
        max_queue_size=config['max_queue_size'],
        use_multiprocessing=False,
        workers=worker_number,
        initial_epoch=initial_epoch,
    )
    return history


def resume_training_model(model_name, row_start=None, row_end=None, epoch_number=1,
                          worker_number=1, verbose=2, custom_objects=None):
    if custom_objects is None:
        custom_objects = custom_metrics
    model = load_model(model_name=model_name, custom_objects=custom_objects, does_compile=True)
    if model is not None:
        epoch = load_epoch_number(model_name=model_name)
        initial_epoch = epoch + 1
    else:
        model = build_model(model_name=model_name)
        compile_model(model=model)
        initial_epoch = 0
    history = train_model(
        model_name=model_name,
        model=model,
        row_start=row_start,
        row_end=row_end,
        epoch_number=epoch_number,
        initial_epoch=initial_epoch,
        worker_number=worker_number,
        verbose=verbose,
    )
    return history


def resume_training_latest_model(row_start=None, row_end=None, epoch_number=1,
                                 worker_number=1, verbose=2, custom_objects=None):
    model_name = get_latest_version_model_name()
    history = resume_training_model(
        model_name=model_name,
        row_start=row_start,
        row_end=row_end,
        epoch_number=epoch_number,
        worker_number=worker_number,
        verbose=verbose,
        custom_objects=custom_objects,
    )
    return history


class UnknownCallbackNameException(ValueError):

    def __init__(self, name):
        super().__init__('Callback name "%s" is unknown.' % name)
