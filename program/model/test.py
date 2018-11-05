from data.data_generator import DataGenerator
from data.dataset_name import DATASET_NAME_TEST
from .metric import custom_metrics
from .save_and_load import load_model
from .version.navi import get_rolling_window_size, get_latest_version_model_name

default_batch_size = 199
default_does_shuffle = False
default_max_queue_size = 10
default_does_use_multiprocessing = False
default_worker_number = 4
default_verbose = 1

config = {
    'batch_size': default_batch_size,
    'does_shuffle': default_does_shuffle,
    'max_queue_size': default_max_queue_size,
    'does_use_multiprocessing': default_does_use_multiprocessing,
    'worker_number': default_worker_number,
    'verbose': default_verbose,
}


def test_model(model_name, model=None, row_start=None, row_end=None, custom_objects=None):
    if 'batch_size' not in config:
        config['batch_size'] = default_batch_size
    if 'does_shuffle' not in config:
        config['does_shuffle'] = default_does_shuffle
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
        dataset_name=DATASET_NAME_TEST,
        rolling_window_size=rolling_window_size,
        row_start=row_start,
        row_end=row_end,
        max_batch_size=config['batch_size'],
        does_shuffle=config['does_shuffle'],
    )
    result_list = model.evaluate_generator(
        generator=generator,
        max_queue_size=config['max_queue_size'],
        use_multiprocessing=config['does_use_multiprocessing'],
        workers=config['worker_number'],
        verbose=config['verbose'],
    )
    result_map = dict(zip(model.metrics_names, result_list))
    return result_map


def test_latest_model(row_start=None, row_end=None, custom_objects=None):
    model_name = get_latest_version_model_name()
    result_map = test_model(
        model_name=model_name,
        row_start=row_start,
        row_end=row_end,
        custom_objects=custom_objects,
    )
    return result_map


class NoTrainedModelException(RuntimeError):

    def __init__(self, model_name):
        super().__init__('Cannot find trained model named "%s".' % model_name)
