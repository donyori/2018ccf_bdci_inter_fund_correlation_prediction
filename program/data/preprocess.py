import numpy as np

from constants import EPSILON
from .combine_data import combination_column_range_map, COMBINATION_COLUMN_RANGE_KEY_FUND_RETURN, \
    COMBINATION_COLUMN_RANGE_KEY_FUND_BENCHMARK_RETURN, COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN, \
    COMBINATION_COLUMN_RANGE_KEY_CORRELATION
from .dataset_name import *
from .load_dataset import load_dataset_np
from .path import TRAIN_PREPROCESSED_DATA_FILE_PATH, TEST_PREPROCESSED_DATA_FILE_PATH, \
    PREDICT_PREPROCESSED_DATA_FILE_PATH

MIN_MAX_KEY_FUND_RETURN = COMBINATION_COLUMN_RANGE_KEY_FUND_RETURN
MIN_MAX_KEY_FUND_BENCHMARK_RETURN = COMBINATION_COLUMN_RANGE_KEY_FUND_BENCHMARK_RETURN
MIN_MAX_KEY_INDEX_RETURN = COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN
MIN_MAX_KEY_CORRELATION = COMBINATION_COLUMN_RANGE_KEY_CORRELATION
MIN_MAX_KEY_TARGET = 'target'

min_max_map = {
    MIN_MAX_KEY_FUND_RETURN: (-0.1, 0.1),
    MIN_MAX_KEY_FUND_BENCHMARK_RETURN: (-0.1, 0.1),
    MIN_MAX_KEY_INDEX_RETURN: (-0.1, 0.1),
    MIN_MAX_KEY_CORRELATION: (-1., 1.),
    MIN_MAX_KEY_TARGET: (0.1, 0.9),  # To avoid saturation Basheer & Najmeer (2000) recommend the range 0.1 and 0.9.
}


def min_max_normalize(dataset, does_overwrite=False):
    if not does_overwrite:
        dataset = dataset.copy()
    inner_size = dataset.shape[-1]
    target_min = min_max_map[MIN_MAX_KEY_TARGET][0]
    target_max = min_max_map[MIN_MAX_KEY_TARGET][1]
    for field_name, column_range in combination_column_range_map.items():
        if column_range[1] - 1 > inner_size:
            continue
        sub_dataset = dataset[:, column_range[0]-1:column_range[1]-1]
        min = min_max_map[field_name][0]
        max = min_max_map[field_name][1]
        max_sub_min = max - min
        if max_sub_min < EPSILON:
            max_sub_min = EPSILON
        factor = (target_max - target_min) / max_sub_min
        if min != 0.:
            sub_dataset -= min
        if factor != 1.:
            sub_dataset *= factor
        if target_min != 0.:
            sub_dataset += target_min
        # Equivalent to: sub_dataset[:] = (sub_dataset - min) / max_sub_min * (target_max - target_min) + target_min
    return dataset


def restore_correlation_from_min_max_normalize(correlation, does_overwrite=False):
    if not does_overwrite:
        correlation = correlation.copy()
    min = min_max_map[MIN_MAX_KEY_TARGET][0]
    max = min_max_map[MIN_MAX_KEY_TARGET][1]
    target_min = min_max_map[MIN_MAX_KEY_CORRELATION][0]
    target_max = min_max_map[MIN_MAX_KEY_CORRELATION][1]
    max_sub_min = max - min
    if max_sub_min < EPSILON:
        max_sub_min = EPSILON
    factor = (target_max - target_min) / max_sub_min
    if min != 0.:
        correlation -= min
    if factor != 1.:
        correlation *= factor
    if target_min != 0.:
        correlation += target_min
    return correlation


def preprocess_data(dataset_name):
    dataset_name = dataset_name.lower()
    dataset = load_dataset_np(dataset_name)
    if dataset_name == DATASET_NAME_TRAIN:
        filename = TRAIN_PREPROCESSED_DATA_FILE_PATH
    elif dataset_name == DATASET_NAME_TEST:
        filename = TEST_PREPROCESSED_DATA_FILE_PATH
    elif dataset_name == DATASET_NAME_PREDICT:
        filename = PREDICT_PREPROCESSED_DATA_FILE_PATH
    else:
        raise UnknownDatasetNameException(dataset_name)
    dataset = min_max_normalize(dataset, does_overwrite=True)
    np.save(file=filename, arr=dataset)
