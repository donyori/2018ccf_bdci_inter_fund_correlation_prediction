import unittest

from .analyze import show_statistics, show_statistics_on_every_fields
from .dataset_name import DATASET_NAME_TRAIN, DATASET_NAME_TEST, DATASET_NAME_PREDICT


class TestAnalyze(unittest.TestCase):

    def test_show_statistics(self):
        show_statistics(DATASET_NAME_TRAIN)
        show_statistics(DATASET_NAME_TEST)
        show_statistics(DATASET_NAME_PREDICT)

    def test_show_statistics_on_every_fields(self):
        show_statistics_on_every_fields(DATASET_NAME_TRAIN)
        show_statistics_on_every_fields(DATASET_NAME_TEST)
        show_statistics_on_every_fields(DATASET_NAME_PREDICT)
