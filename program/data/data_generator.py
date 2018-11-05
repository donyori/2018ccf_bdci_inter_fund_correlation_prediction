import random
from abc import abstractmethod

import numpy as np
from tensorflow import keras

from constants import INDEX_RETURN_INDICATOR_NUMBER
from model.constants import FUND1_RETURN_NAME, FUND1_BENCHMARK_RETURN_NAME, \
    FUND2_RETURN_NAME, FUND2_BENCHMARK_RETURN_NAME, INDEX_RETURN_NAME, MAIN_OUTPUT_NAME, AUXILIARY_OUTPUT_NAME
from .combine_data import COMBINATION_COLUMN_RANGE_KEY_FUND_RETURN, \
    COMBINATION_COLUMN_RANGE_KEY_FUND_BENCHMARK_RETURN, COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN, \
    COMBINATION_COLUMN_RANGE_KEY_CORRELATION, combination_column_range_map
from .dataset_name import DATASET_NAME_PREDICT
from .load_dataset import load_preprocessed_dataset_np
from .preprocess import min_max_map, MIN_MAX_KEY_TARGET
from .util import parse_index, parse_square_ex_index

_frs = combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_FUND_RETURN][0] - 1
_fbrs = combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_FUND_BENCHMARK_RETURN][0] - 1
_irr = tuple(x - 1 for x in combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN])
_cs = combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_CORRELATION][0] - 1

_max_c = np.float32(min_max_map[MIN_MAX_KEY_TARGET][1])


class BaseDataGenerator(keras.utils.Sequence):

    def __init__(self, dataset_name, rolling_window_size, row_start=None, row_end=None,
                 max_batch_size=199, does_shuffle=True):
        if rolling_window_size <= 0:
            raise NonPositiveRollingWindowSizeException(rolling_window_size)
        if max_batch_size <= 0:
            raise NonPositiveBatchSizeException(max_batch_size)
        self._dataset_name = dataset_name.lower()
        self._is_for_prediction = (self._dataset_name == DATASET_NAME_PREDICT.lower())
        self._rolling_window_size = rolling_window_size
        self._batch_size = max_batch_size
        self._does_shuffle = does_shuffle
        self._dataset = load_preprocessed_dataset_np(self._dataset_name)
        if row_start is not None:
            if row_end is not None:
                self._dataset = self._dataset[row_start:row_end]
            else:
                self._dataset = self._dataset[row_start:]
        elif row_end is not None:
            self._dataset = self._dataset[:row_end]
        self._row_number = len(self._dataset)
        if self._rolling_window_size > self._row_number:
            raise RollingWindowSizeTooLargeException(self._rolling_window_size, self._row_number)
        snpr = self.get_sample_number_per_row()
        self._sample_number = (self._row_number - self._rolling_window_size + 1) * snpr
        if self._batch_size is None or self._batch_size > self._sample_number:
            self._batch_size = self._sample_number
        self._index_sequence = list(range(self._sample_number))
        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self._index_sequence[index * self._batch_size: (index+1) * self._batch_size]
        fund1_return = self.__new_empty_array(1)
        fund1_benchmark_return = self.__new_empty_array(1)
        fund2_return = self.__new_empty_array(1)
        fund2_benchmark_return = self.__new_empty_array(1)
        index_return = self.__new_empty_array(INDEX_RETURN_INDICATOR_NUMBER)
        input_dict = {
            FUND1_RETURN_NAME: fund1_return,
            FUND1_BENCHMARK_RETURN_NAME: fund1_benchmark_return,
            FUND2_RETURN_NAME: fund2_return,
            FUND2_BENCHMARK_RETURN_NAME: fund2_benchmark_return,
            INDEX_RETURN_NAME: index_return,
        }
        if not self._is_for_prediction:
            correlation = self.__new_empty_array(None)
            output_dict = {MAIN_OUTPUT_NAME: correlation, AUXILIARY_OUTPUT_NAME: correlation}
        else:
            output_dict = None
        self._feed_batch(indexes=indexes, input_dict=input_dict, output_dict=output_dict)
        if not self._is_for_prediction:
            return input_dict, output_dict
        else:
            return input_dict

    def __len__(self):
        return int(self._sample_number // self._batch_size)

    def on_epoch_end(self):
        if self._does_shuffle:
            random.shuffle(self._index_sequence)

    def get_dataset_name(self):
        return self._dataset_name

    def get_rolling_window_size(self):
        return self._rolling_window_size

    def get_batch_size(self):
        return self._batch_size

    def get_sample_number(self):
        return self._sample_number

    def is_for_prediction(self):
        return self._is_for_prediction

    def _get_sub_dataset(self, start, end):
        return self._dataset[start: end]

    @abstractmethod
    def get_sample_number_per_row(self):
        raise NotImplementedError

    @abstractmethod
    def _feed_batch(self, indexes, input_dict, output_dict):
        raise NotImplementedError

    def __new_empty_array(self, length):
        if length is not None:
            a = np.empty(shape=(self._batch_size, self._rolling_window_size, length), dtype=np.float32)
        else:
            a = np.empty(shape=(self._batch_size, 1), dtype=np.float32)
        return a


class DataGenerator(BaseDataGenerator):

    def __init__(self, dataset_name, rolling_window_size, row_start=None, row_end=None,
                 max_batch_size=199, does_shuffle=True):
        super().__init__(
            dataset_name=dataset_name,
            rolling_window_size=rolling_window_size,
            row_start=row_start,
            row_end=row_end,
            max_batch_size=max_batch_size,
            does_shuffle=does_shuffle
        )

    def get_sample_number_per_row(self):
        return 19900

    def _feed_batch(self, indexes, input_dict, output_dict):
        rws = self.get_rolling_window_size()
        f1r = input_dict[FUND1_RETURN_NAME]
        f1br = input_dict[FUND1_BENCHMARK_RETURN_NAME]
        f2r = input_dict[FUND2_RETURN_NAME]
        f2br = input_dict[FUND2_BENCHMARK_RETURN_NAME]
        ir = input_dict[INDEX_RETURN_NAME]
        c = None
        if output_dict is not None:
            c = output_dict[MAIN_OUTPUT_NAME]
        for i, index in enumerate(indexes, start=0):
            date_seq_no, fund1_no, fund2_no, c_no = parse_index(index)
            date_seq_end_no = date_seq_no + rws
            sub_dataset = self._get_sub_dataset(date_seq_no, date_seq_end_no)
            f1r[i] = np.expand_dims(sub_dataset[:, _frs+fund1_no], axis=-1)
            f1br[i] = np.expand_dims(sub_dataset[:, _fbrs+fund1_no], axis=-1)
            f2r[i] = np.expand_dims(sub_dataset[:, _frs+fund2_no], axis=-1)
            f2br[i] = np.expand_dims(sub_dataset[:, _fbrs+fund2_no], axis=-1)
            ir[i] = sub_dataset[:, _irr[0]:_irr[1]]
            if c is not None:
                c[i][0] = sub_dataset[-1][_cs+c_no]


class SquareExDataGenerator(BaseDataGenerator):

    def __init__(self, dataset_name, rolling_window_size, row_start=None, row_end=None,
                 max_batch_size=200, does_shuffle=True):
        super().__init__(
            dataset_name=dataset_name,
            rolling_window_size=rolling_window_size,
            row_start=row_start,
            row_end=row_end,
            max_batch_size=max_batch_size,
            does_shuffle=does_shuffle,
        )

    def get_sample_number_per_row(self):
        return 40000

    def _feed_batch(self, indexes, input_dict, output_dict):
        rws = self.get_rolling_window_size()
        f1r = input_dict[FUND1_RETURN_NAME]
        f1br = input_dict[FUND1_BENCHMARK_RETURN_NAME]
        f2r = input_dict[FUND2_RETURN_NAME]
        f2br = input_dict[FUND2_BENCHMARK_RETURN_NAME]
        ir = input_dict[INDEX_RETURN_NAME]
        c = None
        if output_dict is not None:
            c = output_dict[MAIN_OUTPUT_NAME]
        for i, index in enumerate(indexes, start=0):
            date_seq_no, fund1_no, fund2_no, c_no = parse_square_ex_index(index)
            date_seq_end_no = date_seq_no + rws
            sub_dataset = self._get_sub_dataset(date_seq_no, date_seq_end_no)
            f1r[i] = np.expand_dims(sub_dataset[:, _frs+fund1_no], axis=-1)
            f1br[i] = np.expand_dims(sub_dataset[:, _fbrs+fund1_no], axis=-1)
            f2r[i] = np.expand_dims(sub_dataset[:, _frs+fund2_no], axis=-1)
            f2br[i] = np.expand_dims(sub_dataset[:, _fbrs+fund2_no], axis=-1)
            ir[i] = sub_dataset[:, _irr[0]:_irr[1]]
            if c is not None:
                if c_no is not None:
                    c[i][0] = sub_dataset[-1][_cs+c_no]
                else:
                    c[i][0] = _max_c


class NonPositiveRollingWindowSizeException(ValueError):

    def __init__(self, rolling_window_size):
        super().__init__('rolling_window_size(%d) is not positive.' % rolling_window_size)


class RollingWindowSizeTooLargeException(ValueError):

    def __init__(self, rolling_window_size, row_number):
        super().__init__('rolling_window_size(%d) is larger than the number of rows(%d).' %
                         (rolling_window_size, row_number))


class NonPositiveBatchSizeException(ValueError):

    def __init__(self, batch_size):
        super().__init__('batch_size(%d) is not positive.' % batch_size)
