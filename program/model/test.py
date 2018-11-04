from data.data_generator import DataGenerator
from data.load_dataset import DATASET_NAME_TEST
from .metric import custom_metrics
from .save_and_load import load_model
from .version.navi import get_rolling_window_size, get_latest_version_model_name

default_batch_size = 199
default_max_queue_size = 10

config = {
    'batch_size': default_batch_size,
    'max_queue_size': default_max_queue_size,
}


def test_model(model_name, model=None, row_start=None, row_end=None, worker_number=1, verbose=1, custom_objects=None):
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
        dataset_name=DATASET_NAME_TEST,
        rolling_window_size=rolling_window_size,
        row_start=row_start,
        row_end=row_end,
        max_batch_size=config['batch_size'],
        does_shuffle=True,
    )
    result_list = model.evaluate_generator(
        generator=generator,
        max_queue_size=config['max_queue_size'],
        use_multiprocessing=False,
        workers=worker_number,
        verbose=verbose,
    )
    result_map = dict(zip(model.metrics_names, result_list))
    return result_map


def test_latest_model(row_start=None, row_end=None, worker_number=1, verbose=1, custom_objects=None):
    model_name = get_latest_version_model_name()
    result_map = test_model(
        model_name=model_name,
        row_start=row_start,
        row_end=row_end,
        worker_number=worker_number,
        verbose=verbose,
        custom_objects=custom_objects,
    )
    return result_map


class NoTrainedModelException(RuntimeError):

    def __init__(self, model_name):
        super().__init__('Cannot find trained model named "%s".' % model_name)
