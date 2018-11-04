import unittest

from .util import _first_index_in_every_row_list, parse_index, calculate_correlation_no

_pair_list = [(i, j) for i in range(199) for j in range(i+1, 200)]
_index_map = dict(zip(_pair_list, range(19900)))
_reverse_index_map = dict(zip(range(19900), _pair_list))


class TestUtil(unittest.TestCase):

    def test_first_index_in_every_row(self):
        print(_first_index_in_every_row_list)
        self.assertEqual(len(_first_index_in_every_row_list), 199)
        for row in range(199):
            self.assertEqual(_index_map[(row, row+1)], _first_index_in_every_row_list[row])

    def test_parse_index(self):
        for index in range(19900):
            _, fund1_no, fund2_no, correlation_no = parse_index(index)
            self.assertEqual(correlation_no, index)
            self.assertEqual((fund1_no, fund2_no), _reverse_index_map[index])

    def test_calculate_correlation_no(self):
        for fund1_no in range(199):
            for fund2_no in range(fund1_no+1, 200):
                correlation_no1 = calculate_correlation_no(fund1_no, fund2_no)
                correlation_no2 = calculate_correlation_no(fund2_no, fund1_no)
                self.assertEqual(correlation_no1, correlation_no2)
                self.assertEqual(correlation_no1, _index_map[(fund1_no, fund2_no)],
                                 msg='f1=%d, f2=%d, c=%d!=%d' % (
                                     fund1_no, fund2_no, correlation_no1, _index_map[(fund1_no, fund2_no)]))
