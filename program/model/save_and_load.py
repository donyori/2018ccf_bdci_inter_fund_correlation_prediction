import os
import pathlib
import threading
import time

from tensorflow import keras

from constants import PROJECT_HOME
from .version.navi import get_latest_version_model_name

MODEL_HOME = os.path.join(PROJECT_HOME, 'model')

_latest_filename = 'latest'
_latest_epoch_number_filename = 'latest_epoch_number'
_global_lock = threading.Lock()


def save_model(model, model_name, does_overwrite=True, does_include_optimizer=False):
    directory = _get_and_ensure_directory(model_name)
    timestamp = str(time.time())
    filename = os.path.join(directory, '%s.%s.h5' % (model_name, timestamp))
    model.save(filename, overwrite=does_overwrite, include_optimizer=does_include_optimizer)
    _set_latest_filename(model_name=model_name, filename=filename)


def save_latest_model(model, does_overwrite=True, does_include_optimizer=False):
    save_model(model=model, model_name=get_latest_version_model_name(),
               does_overwrite=does_overwrite, does_include_optimizer=does_include_optimizer)


def load_model(model_name, custom_objects=None, does_compile=True):
    filename = _get_latest_filename(model_name=model_name)
    if not filename:
        return None
    model = keras.models.load_model(filename, custom_objects=custom_objects, compile=does_compile)
    return model


def load_latest_model(custom_objects=None, does_compile=True):
    return load_model(model_name=get_latest_version_model_name(),
                      custom_objects=custom_objects,
                      does_compile=does_compile)


def save_epoch_number(model_name, epoch):
    directory = _get_and_ensure_directory(model_name)
    filename = os.path.join(directory, _latest_epoch_number_filename)
    with open(filename, mode='w') as f:
        f.write(str(epoch))


def save_latest_model_epoch_number(epoch):
    save_epoch_number(model_name=get_latest_version_model_name(), epoch=epoch)


def load_epoch_number(model_name):
    try:
        filename = os.path.join(MODEL_HOME, model_name.replace('.', '_'), _latest_epoch_number_filename)
        with open(filename, newline='') as f:
            content = ''.join(f.readlines()).strip()
            epoch = int(content)
            return epoch
    except FileNotFoundError:
        return -1


def load_latest_model_epoch_number():
    return load_epoch_number(get_latest_version_model_name())


def _get_and_ensure_directory(model_name):
    directory = os.path.join(MODEL_HOME, model_name.replace('.', '_'))
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def _get_latest_filename(model_name):
    _global_lock.acquire(blocking=True)
    try:
        latest_filename = os.path.join(MODEL_HOME, model_name.replace('.', '_'), _latest_filename)
        with open(latest_filename, newline='') as f:
            content = ''.join(f.readlines()).strip()
            content = os.path.basename(content)
            directory = _get_and_ensure_directory(model_name)
            filename = os.path.join(directory, content)
            return filename
    except FileNotFoundError:
        return ''
    finally:
        _global_lock.release()


def _set_latest_filename(model_name, filename):
    _global_lock.acquire(blocking=True)
    try:
        directory = _get_and_ensure_directory(model_name)
        latest_filename = os.path.join(directory, _latest_filename)
        with open(latest_filename, mode='w') as f:
            f.write(os.path.basename(filename))
    finally:
        _global_lock.release()
