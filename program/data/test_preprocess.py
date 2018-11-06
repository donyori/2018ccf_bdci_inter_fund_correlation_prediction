import unittest

import numpy as np

from .analyze import show_statistics_on_every_fields
from .combine_data import combination_column_range_map, COMBINATION_COLUMN_RANGE_KEY_CORRELATION
from .dataset_name import DATASET_NAME_TRAIN, DATASET_NAME_TEST, DATASET_NAME_PREDICT
from .load_dataset import load_dataset_np
from .preprocess import min_max_normalize, restore_correlation_from_min_max_normalize


class TestPreprocess(unittest.TestCase):

    def test_min_max_normalize(self):
        self.__test_normalize_sub_process(DATASET_NAME_TRAIN)
        self.__test_normalize_sub_process(DATASET_NAME_TEST)
        self.__test_normalize_sub_process(DATASET_NAME_PREDICT)

    def test_restore_correlation_from_min_max_normalize(self):
        crr = tuple(x - 1 for x in combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_CORRELATION])
        dataset = load_dataset_np(DATASET_NAME_TEST)
        correlation = dataset[:, crr[0]:crr[1]]
        ds = min_max_normalize(dataset, does_overwrite=False)
        c = ds[:, crr[0]:crr[1]]
        c = restore_correlation_from_min_max_normalize(c, does_overwrite=False)
        self.assertEqual(np.allclose(correlation, c, rtol=0., atol=1e-4), True)

    def __test_normalize_sub_process(self, dataset_name):
        print('Dataset:', dataset_name)
        dataset = load_dataset_np(dataset_name)
        ds = min_max_normalize(dataset, does_overwrite=False)
        self.assertEqual(np.array_equal(dataset, ds), False)
        ds = min_max_normalize(dataset, does_overwrite=True)
        self.assertEqual(np.array_equal(dataset, ds), True)
        show_statistics_on_every_fields(ds)
