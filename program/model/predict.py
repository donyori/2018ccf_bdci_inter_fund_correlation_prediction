from data.data_generator import DataGenerator
from data.load_dataset import DATASET_NAME_PREDICT
from .metric import custom_metrics
from .save_and_load import load_model
from .test import NoTrainedModelException
from .version.navi import get_rolling_window_size, get_latest_version_model_name

default_batch_size = 199
default_max_queue_size = 10

config = {
    'batch_size': default_batch_size,
    'max_queue_size': default_max_queue_size,
}


def predict(model_name, model=None, row_start=None, row_end=None, worker_number=1, verbose=1, custom_objects=None):
    if 'batch_size' not in config:
        config['batch_size'] = default_batch_size
    if 'max_queue_size' not in config:
        config['max_queue_size'] = default_max_queue_size
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
        use_multiprocessing=False,
        workers=worker_number,
        verbose=verbose,
    )
    return result


def predict_last_row(model_name, worker_number=1, verbose=1, custom_objects=None):
    rolling_window_size = get_rolling_window_size(model_name=model_name)
    row_start = -rolling_window_size
    result = predict(
        model_name=model_name,
        row_start=row_start,
        worker_number=worker_number,
        verbose=verbose,
        custom_objects=custom_objects,
    )
    return result


def predict_using_latest_model(row_start=None, row_end=None, worker_number=1, verbose=1, custom_objects=None):
    model_name = get_latest_version_model_name()
    result = predict(
        model_name=model_name,
        row_start=row_start,
        row_end=row_end,
        worker_number=worker_number,
        verbose=verbose,
        custom_objects=custom_objects,
    )
    return result


def predict_last_row_using_latest_model(worker_number=1, verbose=1, custom_objects=None):
    model_name = get_latest_version_model_name()
    result = predict_last_row(
        model_name=model_name,
        worker_number=worker_number,
        verbose=verbose,
        custom_objects=custom_objects,
    )
    return result
