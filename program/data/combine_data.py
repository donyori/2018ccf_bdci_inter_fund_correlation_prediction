import csv

from .date_number_map import date_number_map
from .path import *

COMBINATION_COLUMN_RANGE_KEY_FUND_RETURN = 'fund_return'
COMBINATION_COLUMN_RANGE_KEY_FUND_BENCHMARK_RETURN = 'fund_benchmark_return'
COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN = 'index_return'
COMBINATION_COLUMN_RANGE_KEY_CORRELATION = 'correlation'

combination_column_range_map = {
    COMBINATION_COLUMN_RANGE_KEY_FUND_RETURN: (1, 201),
    COMBINATION_COLUMN_RANGE_KEY_FUND_BENCHMARK_RETURN: (201, 401),
    COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN: (401, 436),
    COMBINATION_COLUMN_RANGE_KEY_CORRELATION: (436, 20336)
}

COMBINATION_COLUMN_NUMBER = combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_CORRELATION][-1]


def combine_data(is_train):
    if is_train:
        fund_return = TRAIN_FUND_RETURN_FILE_PATH
        fund_benchmark_return = TRAIN_FUND_BENCHMARK_RETURN_FILE_PATH
        index_return = TRAIN_INDEX_RETURN_FILE_PATH
        correlation = TRAIN_CORRELATION_FILE_PATH
        combination = TRAIN_COMBINATION_FILE_PATH
    else:
        fund_return = TEST_FUND_RETURN_FILE_PATH
        fund_benchmark_return = TEST_FUND_BENCHMARK_RETURN_FILE_PATH
        index_return = TEST_INDEX_RETURN_FILE_PATH
        correlation = TEST_CORRELATION_FILE_PATH
        combination = TEST_COMBINATION_FILE_PATH
    appeared_date_no = set()

    def read_csv_into_dict(filename, date_no_offset=0):
        result = dict()
        with open(filename, newline='') as f:
            item_names = list()
            reader = csv.DictReader(f)
            for row in reader:
                name = row['']
                item_names.append(name)
                for k, v in row.items():
                    if not k:
                        continue
                    date_no = date_number_map[k] + date_no_offset
                    if date_no not in result:
                        result[date_no] = dict()
                    if name in result[date_no]:
                        raise DuplicatedItemException(name, filename)
                    result[date_no][name] = v
                    appeared_date_no.add(date_no)
        return result, item_names

    fund_return_map, _ = read_csv_into_dict(fund_return)
    fund_benchmark_return_map, _ = read_csv_into_dict(fund_benchmark_return)
    index_return_map, index_return_item_names = read_csv_into_dict(index_return)
    correlation_map, _ = read_csv_into_dict(correlation, date_no_offset=-1)

    appeared_date_no_list = list(appeared_date_no)
    appeared_date_no_list.sort()

    def check_map_size(m, size, filename):
        if len(m) != size:
            raise UnexpectedFileException(filename)

    headers = ['dn'] + ['f%d_fr' % i for i in range(200)] + \
              ['f%d_fbr' % i for i in range(200)] + \
              ['i%d' % i for i in range(35)] + \
              ['f%d_f%d' % (i, j) for i in range(199) for j in range(i+1, 200)]
    with open(combination, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for date_no in appeared_date_no_list:
            row = list()
            row.append(date_no)
            fr = fund_return_map[date_no]
            check_map_size(fr, 200, fund_return)
            for i in range(200):
                row.append(fr['Fund %d' % (i+1)])
            fbr = fund_benchmark_return_map[date_no]
            check_map_size(fbr, 200, fund_benchmark_return)
            for i in range(200):
                row.append(fbr['Fund %d' % (i+1)])
            ir = index_return_map[date_no]
            check_map_size(ir, 35, index_return)
            for name in index_return_item_names:
                row.append(ir[name])
            # correlation.csv may not have whole data.
            if date_no in correlation_map:
                c = correlation_map[date_no]
                check_map_size(c, 19900, correlation)
                for i in range(199):
                    for j in range(i + 1, 200):
                        row.append(c['Fund %d-Fund %d' % (i + 1, j + 1)])
            else:
                row.extend([''] * 19900)
            writer.writerow(row)


class DuplicatedItemException(RuntimeError):

    def __init__(self, item_name, filename):
        super().__init__('Duplicated item "%s" in the file "%s".' % (item_name, filename))


class UnexpectedFileException(RuntimeError):

    def __init__(self, filename):
        super().__init__('File "%s" is unexpected. Perhaps it is corrupt.' % filename)
