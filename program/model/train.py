import os
import re

from tensorflow import keras

from constants import PROJECT_HOME, EPSILON
from data.data_generator import SquareExDataGenerator
from data.dataset_name import DATASET_NAME_TRAIN
from util import str_to_bool
from .callback import ModelSaver, EpochNumberSaver, TimeLimiter
from .compile import compile_model
from .constants import MAIN_OUTPUT_NAME
from .metric import custom_metrics
from .save_and_load import load_model, load_epoch_number
from .version.navi import get_rolling_window_size, build_model, get_latest_version_model_name

LOG_DIRECTORY = os.path.join(PROJECT_HOME, 'log')

_remove_pattern = re.compile(r'[\s|_]+', flags=re.UNICODE)

default_batch_size = 200
default_does_shuffle = True
default_callbacks = ['model_saver', 'epoch_number_saver', 'tensor_board', 'early_stopping']
default_max_queue_size = 10
default_does_use_multiprocessing = False
default_worker_number = 4
default_verbose = 2

config = {
    'batch_size': default_batch_size,
    'does_shuffle': default_does_shuffle,
    'callbacks': default_callbacks,
    'max_queue_size': default_max_queue_size,
    'does_use_multiprocessing': default_does_use_multiprocessing,
    'worker_number': default_worker_number,
    'verbose': default_verbose,
}


def train_model(model_name, model, row_start=None, row_end=None, initial_epoch=0, end_epoch=1, time_limit=None):
    if initial_epoch >= end_epoch:
        print('initial_epoch(%d) >= end_epoch(%d).')
        return None
    if 'batch_size' not in config:
        config['batch_size'] = default_batch_size
    if 'does_shuffle' not in config:
        config['does_shuffle'] = default_does_shuffle
    if 'callbacks' not in config:
        config['callbacks'] = default_callbacks
    if 'max_queue_size' not in config:
        config['max_queue_size'] = default_max_queue_size
    if 'does_use_multiprocessing' not in config:
        config['does_use_multiprocessing'] = default_does_use_multiprocessing
    if 'worker_number' not in config:
        config['worker_number'] = default_worker_number
    if 'verbose' not in config:
        config['verbose'] = default_verbose
    callbacks = list() if config['callbacks'] is not None else None
    if callbacks is not None:
        for cb in config['callbacks']:
            if isinstance(cb, keras.callbacks.Callback):
                if isinstance(cb, TimeLimiter) and time_limit is not None:
                    print('WARNING: time_limit is not None, ignored TimeLimiter in config.')
                    continue
                callbacks.append(cb)
            elif isinstance(cb, str):
                cb_str = cb.lower()
                cb_str = re.sub(pattern=_remove_pattern, repl='', string=cb_str)
                sep_idx = cb_str.find(':')
                cb_params = dict()
                if sep_idx >= 0:
                    cb_name = cb_str[:sep_idx]
                    cb_params_strs = cb_str[sep_idx+1:].split(',')
                    for cb_param_str in cb_params_strs:
                        eq_idx = cb_param_str.find('=')
                        if eq_idx >= 0:
                            cb_params[cb_param_str[:eq_idx]] = cb_param_str[eq_idx+1:]
                        else:
                            cb_params[cb_param_str] = '1'
                else:
                    cb_name = cb_str
                if cb_name == 'earlystopping':
                    callbacks.append(keras.callbacks.EarlyStopping(
                        monitor=MAIN_OUTPUT_NAME+'_loss' if 'monitor' not in cb_params else cb_params['monitor'],
                        min_delta=EPSILON if 'min_delta' not in cb_params else float(cb_params['min_delta']),
                        patience=3 if 'patience' not in cb_params else int(cb_params['patience']),
                        verbose=1 if 'verbose' not in cb_params else int(cb_params['verbose']),
                    ))
                elif cb_name == 'tensorboard':
                    callbacks.append(keras.callbacks.TensorBoard(
                        log_dir=os.path.join(LOG_DIRECTORY, model_name)
                        if 'log_dir' not in cb_params else cb_params['log_dir'],
                        batch_size=config['batch_size'],
                        write_graph=True if 'write_graph' not in cb_params else str_to_bool(cb_params['write_graph']),
                    ))
                elif cb_name == 'modelsaver':
                    callbacks.append(ModelSaver(
                        model_name=model_name,
                        period=1 if 'period' not in cb_params else int(cb_params['period']),
                        verbose=1 if 'verbose' not in cb_params else int(cb_params['verbose']),
                    ))
                elif cb_name == 'epochnumbersaver':
                    callbacks.append(EpochNumberSaver(
                        model_name=model_name,
                        verbose=1 if 'verbose' not in cb_params else int(cb_params['verbose']),
                    ))
                elif cb_name == 'timelimiter':
                    if time_limit is not None:
                        print('WARNING: time_limit is not None, ignored TimeLimiter in config.')
                        continue
                    if 'limit' not in cb_params:
                        raise ValueError("TimeLimiter's parameter limit is missed.")
                    callbacks.append(TimeLimiter(
                        limit=cb_params['limit'],
                        verbose=1 if 'verbose' not in cb_params else int(cb_params['verbose']),
                    ))
                else:
                    raise UnknownCallbackNameException(cb)
            else:
                raise TypeError('Callback must be an instance of keras.callbacks.Callback or a callback name(string).')
    if time_limit is not None:
        callbacks.append(TimeLimiter(limit=time_limit, verbose=1))
    rolling_window_size = get_rolling_window_size(model_name)
    generator = SquareExDataGenerator(
        dataset_name=DATASET_NAME_TRAIN,
        rolling_window_size=rolling_window_size,
        row_start=row_start,
        row_end=row_end,
        max_batch_size=config['batch_size'],
        does_shuffle=config['does_shuffle'],
    )
    history = model.fit_generator(
        generator=generator,
        epochs=end_epoch,
        verbose=config['verbose'],
        callbacks=callbacks,
        max_queue_size=config['max_queue_size'],
        use_multiprocessing=config['does_use_multiprocessing'],
        workers=config['worker_number'],
        initial_epoch=initial_epoch,
    )
    return history


def resume_training_model(model_name, row_start=None, row_end=None, end_epoch=1, time_limit=None, custom_objects=None):
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
        initial_epoch=initial_epoch,
        end_epoch=end_epoch,
        time_limit=time_limit,
    )
    return history


def resume_training_latest_model(row_start=None, row_end=None, end_epoch=1, time_limit=None, custom_objects=None):
    model_name = get_latest_version_model_name()
    history = resume_training_model(
        model_name=model_name,
        row_start=row_start,
        row_end=row_end,
        end_epoch=end_epoch,
        time_limit=time_limit,
        custom_objects=custom_objects,
    )
    return history


class UnknownCallbackNameException(ValueError):

    def __init__(self, name):
        super().__init__('Callback name "%s" is unknown.' % name)
