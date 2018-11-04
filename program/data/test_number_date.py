import unittest

from .path import TRADING_DATE_FILE_PATH
from .number_date import number_date


class TestNumberDate(unittest.TestCase):

    def test_number_date(self):
        date_map = number_date()
        self.assertNotIn('', date_map)
        with open(TRADING_DATE_FILE_PATH) as file:
            content = file.readlines()
            for i in range(len(content)):
                self.assertEqual(date_map[content[i].strip()], i)
            print('E.g. "2015-09-16" ->', date_map['2015-09-16'])
