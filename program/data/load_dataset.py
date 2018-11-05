import numpy as np

from .combine_data import COMBINATION_COLUMN_NUMBER, \
    combination_column_range_map, COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN
from .dataset_name import *
from .date_range import TEST_CORRELATION_DATE_RANGE
from .path import TRAIN_COMBINATION_FILE_PATH, TEST_COMBINATION_FILE_PATH, \
    TRAIN_PREPROCESSED_DATA_FILE_PATH, TEST_PREPROCESSED_DATA_FILE_PATH, PREDICT_PREPROCESSED_DATA_FILE_PATH


def load_dataset_np(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == DATASET_NAME_TRAIN:
        dataset = np.genfromtxt(
            fname=TRAIN_COMBINATION_FILE_PATH,
            dtype=np.float32,
            delimiter=',',
            skip_header=1,
            filling_values='0.0',
            usecols=range(1, COMBINATION_COLUMN_NUMBER),
            invalid_raise=True
        )
    elif dataset_name == DATASET_NAME_TEST:
        # import date_number_map here because it costs a lot.
        from .date_number_map import date_number_map
        start_date_number = date_number_map[TEST_CORRELATION_DATE_RANGE[0]]
        end_date_number = date_number_map[TEST_CORRELATION_DATE_RANGE[1]]
        dataset = np.genfromtxt(
            fname=TEST_COMBINATION_FILE_PATH,
            dtype=np.float32,
            delimiter=',',
            skip_header=1,
            filling_values='0.0',
            usecols=range(1, COMBINATION_COLUMN_NUMBER),
            invalid_raise=True,
            max_rows=end_date_number - start_date_number + 1
        )
    elif dataset_name == DATASET_NAME_PREDICT:
        # import date_number_map here because it costs a lot.
        from .date_number_map import date_number_map
        start_date_number = date_number_map[TEST_CORRELATION_DATE_RANGE[0]]
        end_date_number = date_number_map[TEST_CORRELATION_DATE_RANGE[1]]
        end_column_no = combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN][-1]
        dataset = np.genfromtxt(
            fname=TEST_COMBINATION_FILE_PATH,
            dtype=np.float32,
            delimiter=',',
            skip_header=2+end_date_number-start_date_number,
            filling_values='0.0',
            usecols=range(1, end_column_no),
            invalid_raise=True
        )
    else:
        raise UnknownDatasetNameException(dataset_name)
    return dataset


def load_preprocessed_dataset_np(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == DATASET_NAME_TRAIN:
        dataset = np.load(file=TRAIN_PREPROCESSED_DATA_FILE_PATH)
    elif dataset_name == DATASET_NAME_TEST:
        dataset = np.load(file=TEST_PREPROCESSED_DATA_FILE_PATH)
    elif dataset_name == DATASET_NAME_PREDICT:
        dataset = np.load(file=PREDICT_PREPROCESSED_DATA_FILE_PATH)
    else:
        raise UnknownDatasetNameException(dataset_name)
    return dataset
