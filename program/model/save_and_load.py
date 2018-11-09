import json
import os
import pathlib
import threading
import time

from tensorflow import keras

from constants import PROJECT_HOME
from .version.navi import get_latest_version_model_name

MODEL_HOME = os.path.join(PROJECT_HOME, 'model')

_global_lock = threading.Lock()


def save_model(model_name, model, epoch=None, does_overwrite=True, does_include_optimizer=False):
    if model is None:
        raise ValueError('model is None.')
    directory = _get_and_ensure_directory(model_name=model_name)
    if epoch is not None:
        filename = os.path.join(directory, '%s.epoch%05d.h5' % (model_name, epoch))
    else:
        filename = os.path.join(directory, '%s.%s.h5' % (model_name, time.time()))
    model.save(filename, overwrite=does_overwrite, include_optimizer=does_include_optimizer)
    _set_latest_filename(model_name=model_name, filename=filename)


def save_latest_model(model, epoch=None, does_overwrite=True, does_include_optimizer=False):
    save_model(model_name=get_latest_version_model_name(), model=model, epoch=epoch,
               does_overwrite=does_overwrite, does_include_optimizer=does_include_optimizer)


def load_model(model_name, epoch=None, custom_objects=None, does_compile=True):
    if epoch is not None:
        directory = _get_directory(model_name=model_name)
        filename = os.path.join(directory, '%s.epoch%05d.h5' % (model_name, epoch))
    else:
        filename = _get_latest_filename(model_name=model_name)
    if not filename:
        return None
    if not os.path.isfile(filename):  # If file does NOT exist:
        return None
    model = keras.models.load_model(filename, custom_objects=custom_objects, compile=does_compile)
    return model


def load_latest_model(epoch=None, custom_objects=None, does_compile=True):
    return load_model(model_name=get_latest_version_model_name(), epoch=epoch,
                      custom_objects=custom_objects, does_compile=does_compile)


def save_variable(model_name, key, value):
    if not isinstance(key, str):
        raise TypeError('key should be str.')
    variables = _load_variables(model_name=model_name)
    if variables is None:
        variables = dict()
    variables[key] = value
    _save_variables(model_name=model_name, variables=variables)


def load_variable(model_name, key):
    if not isinstance(key, str):
        raise TypeError('key should be str.')
    variables = _load_variables(model_name=model_name)
    if variables is None or key not in variables:
        return None
    return variables[key]


def save_last_epoch_number(model_name, epoch):
    save_variable(model_name=model_name, key='last_epoch', value=epoch)


def load_last_epoch_number(model_name):
    epoch = load_variable(model_name=model_name, key='last_epoch')
    if epoch is None:
        epoch = -1
    return epoch


def save_best_info(model_name, epoch, monitor_value, monitor_name='loss'):
    if not monitor_name:
        monitor_name = 'loss'
    info = {'epoch': epoch, monitor_name: monitor_value}
    save_variable(model_name=model_name, key='best', value=info)


def load_best_info(model_name, monitor_name='loss'):
    if not monitor_name:
        monitor_name = 'loss'
    info = load_variable(model_name=model_name, key='best')
    if info is None:
        return None, None
    return info['epoch'], info[monitor_name]


def _get_directory(model_name):
    directory = os.path.join(MODEL_HOME, model_name.replace('.', '_'))
    return directory


def _get_and_ensure_directory(model_name):
    directory = _get_directory(model_name=model_name)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def _save_variables(model_name, variables):
    if not isinstance(variables, dict):
        raise TypeError('variables should be dict.')
    global _global_lock
    _global_lock.acquire(blocking=True)
    try:
        directory = _get_and_ensure_directory(model_name=model_name)
        filename = os.path.join(directory, 'variables.json')
        with open(filename, mode='w', encoding='utf-8') as f:
            json.dump(variables, f, skipkeys=False, ensure_ascii=False, indent=4, sort_keys=True)
    finally:
        _global_lock.release()


def _load_variables(model_name):
    global _global_lock
    _global_lock.acquire(blocking=True)
    try:
        directory = _get_directory(model_name=model_name)
        filename = os.path.join(directory, 'variables.json')
        with open(filename, encoding='utf-8') as f:
            variables = json.load(f)
            if not isinstance(variables, dict):
                raise RuntimeError('Cannot load variables from file.')
            return variables
    except FileNotFoundError:
        return None
    finally:
        _global_lock.release()


def _get_latest_filename(model_name):
    filename = load_variable(model_name=model_name, key='latest_filename')
    if filename is not None:
        filename = filename.strip()
    if not filename:
        return None
    filename = os.path.basename(filename)
    directory = _get_and_ensure_directory(model_name=model_name)
    filename = os.path.join(directory, filename)
    return filename


def _set_latest_filename(model_name, filename):
    filename = os.path.basename(filename)
    save_variable(model_name=model_name, key='latest_filename', value=filename)
