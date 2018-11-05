from data.data_generator import DataGenerator
from data.dataset_name import DATASET_NAME_PREDICT
from .metric import custom_metrics
from .save_and_load import load_model
from .test import NoTrainedModelException
from .version.navi import get_rolling_window_size, get_latest_version_model_name

default_batch_size = 199
default_max_queue_size = 10
default_does_use_multiprocessing = False
default_worker_number = 4
default_verbose = 1

config = {
    'batch_size': default_batch_size,
    'max_queue_size': default_max_queue_size,
    'does_use_multiprocessing': default_does_use_multiprocessing,
    'worker_number': default_worker_number,
    'verbose': default_verbose,
}


def predict(model_name, model=None, row_start=None, row_end=None, custom_objects=None):
    if 'batch_size' not in config:
        config['batch_size'] = default_batch_size
    if 'max_queue_size' not in config:
        config['max_queue_size'] = default_max_queue_size
    if 'does_use_multiprocessing' not in config:
        config['does_use_multiprocessing'] = default_does_use_multiprocessing
    if 'worker_number' not in config:
        config['worker_number'] = default_worker_number
    if 'verbose' not in config:
        config['verbose'] = default_verbose
    if model is None:
        if custom_objects is None:
            custom_objects = custom_metrics
        model = load_model(model_name=model_name, custom_objects=custom_objects, does_compile=True)
        if model is None:
            raise NoTrainedModelException(model_name)
    rolling_window_size = get_rolling_window_size(model_name)
    generator = DataGenerator(
        dataset_name=DATASET_NAME_PREDICT,
        rolling_window_size=rolling_window_size,
        row_start=row_start,
        row_end=row_end,
        max_batch_size=config['batch_size'],
        does_shuffle=False,  # NOT shuffle!
    )
    snpr = generator.get_sample_number_per_row()
    if config['batch_size'] % snpr != 0:
        print('WARNING: batch_size(%d) cannot divide %d. Some inputs will be ignored.' % (config['batch_size'], snpr))

    result = model.predict_generator(
        generator=generator,
        max_queue_size=config['max_queue_size'],
        use_multiprocessing=config['does_use_multiprocessing'],
        workers=config['worker_number'],
        verbose=config['verbose'],
    )
    return result


def predict_last_row(model_name, custom_objects=None):
    rolling_window_size = get_rolling_window_size(model_name=model_name)
    row_start = -rolling_window_size
    result = predict(
        model_name=model_name,
        row_start=row_start,
        custom_objects=custom_objects,
    )
    return result


def predict_using_latest_model(row_start=None, row_end=None, custom_objects=None):
    model_name = get_latest_version_model_name()
    result = predict(
        model_name=model_name,
        row_start=row_start,
        row_end=row_end,
        custom_objects=custom_objects,
    )
    return result


def predict_last_row_using_latest_model(custom_objects=None):
    model_name = get_latest_version_model_name()
    result = predict_last_row(
        model_name=model_name,
        custom_objects=custom_objects,
    )
    return result
