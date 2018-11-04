import csv
import unittest

import numpy as np

from .date_range import TEST_CORRELATION_DATE_RANGE
from .load_dataset import load_dataset_np, load_preprocessed_dataset_np, \
    DATASET_NAME_TRAIN, DATASET_NAME_TEST, DATASET_NAME_PREDICT
from .path import TRAIN_COMBINATION_FILE_PATH, TEST_COMBINATION_FILE_PATH
from .preprocess import min_max_normalize


class TestLoadDataset(unittest.TestCase):

    def test_load_train_dataset_np(self):
        data = load_dataset_np(dataset_name=DATASET_NAME_TRAIN)
        print('size:', data.size, 'dims:', data.ndim, 'shape:', data.shape, 'len():', len(data))
        print(data[0])
        print(data[-1])
        with open(TRAIN_COMBINATION_FILE_PATH, newline='') as f:
            reader = csv.reader(f)
            i = 0
            is_first = True
            for row in reader:
                if is_first:
                    is_first = False
                    continue
                j = 0
                for value in row[1:]:
                    self.assertAlmostEqual(float(value), data[i][j])
                    j += 1
                i += 1

    def test_load_test_dataset_np(self):
        # import date_number_map here because it costs a lot.
        from .date_number_map import date_number_map
        data = load_dataset_np(dataset_name=DATASET_NAME_TEST)
        print('size:', data.size, 'dims:', data.ndim, 'shape:', data.shape, 'len():', len(data))
        print(data[0])
        print(data[-1])
        start_dn = date_number_map[TEST_CORRELATION_DATE_RANGE[0]] - 1
        end_dn = date_number_map[TEST_CORRELATION_DATE_RANGE[1]] - 1
        with open(TEST_COMBINATION_FILE_PATH, newline='') as f:
            reader = csv.reader(f)
            i = 0
            is_first = True
            for row in reader:
                if is_first:
                    is_first = False
                    continue
                dn = int(row[0])
                if dn < start_dn:
                    continue
                if dn > end_dn:
                    break
                j = 0
                for value in row[1:]:
                    self.assertAlmostEqual(float(value), data[i][j])
                    j += 1
                i += 1

    def test_load_preprocessed_dataset_np(self):
        dataset = load_preprocessed_dataset_np(DATASET_NAME_TEST)
        ds = load_dataset_np(DATASET_NAME_TEST)
        ds = min_max_normalize(ds, does_overwrite=True)
        self.assertEqual(np.allclose(dataset, ds), True)
        dataset = load_preprocessed_dataset_np(DATASET_NAME_PREDICT)
        ds = load_dataset_np(DATASET_NAME_PREDICT)
        ds = min_max_normalize(ds, does_overwrite=True)
        self.assertEqual(np.allclose(dataset, ds), True)
