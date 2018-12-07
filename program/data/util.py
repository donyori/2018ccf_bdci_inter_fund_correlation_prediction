_first_index_in_every_row_list = list()


def _build_first_index_in_every_row_list():
    global _first_index_in_every_row_list
    _first_index_in_every_row_list.clear()
    _first_index_in_every_row_list.append(0)
    for delta in range(199, 1, -1):
        _first_index_in_every_row_list.append(_first_index_in_every_row_list[-1] + delta)


_build_first_index_in_every_row_list()


def parse_index(index, step=1):
    if not isinstance(index, int):
        index = int(index)
    date_seq_no = (index // 19900) * step
    correlation_no = index % 19900
    # Use binary search to get fund number:
    # FIXME: Consider to compute fund number directly.
    low = correlation_no // 199                                   # include
    high = min(low * 2 + 1, len(_first_index_in_every_row_list))  # exclude
    while low < high:
        middle = (low + high) // 2
        if _first_index_in_every_row_list[middle] < correlation_no:
            low = middle + 1
        elif _first_index_in_every_row_list[middle] > correlation_no:
            high = middle
        else:
            low = middle
            break
    if _first_index_in_every_row_list[low] > correlation_no:
        low -= 1
    fund1_no = low
    fund2_no = correlation_no - _first_index_in_every_row_list[fund1_no] + fund1_no + 1
    return date_seq_no, fund1_no, fund2_no, correlation_no


def calculate_correlation_no(fund1_no, fund2_no):
    if fund1_no == fund2_no:
        return None
    if fund1_no > fund2_no:
        tmp = fund1_no
        fund1_no = fund2_no
        fund2_no = tmp
    if fund1_no < 0:
        raise ValueError('fund1_no should >= 0, got %d.' % fund1_no)
    if fund2_no >= 200:
        raise ValueError('fund2_no should < 200, got %d.' % fund2_no)
    '''
    input:
    f1 in [0, 198]
    f2 in [f1 + 1, 199]

    output:
    c = (199 + 198 + ... + (199 - f1 + 1)) + (f2 - (f1 + 1))
        |----------- f1 terms -----------|
      = (199 + (199 - f1 + 1)) * f1 / 2 + (f2 - f1 - 1)
      = (399 - f1) * f1 / 2 + f2 - f1 - 1
    '''
    correlation_no = int(((399 - fund1_no) * fund1_no) // 2) + fund2_no - fund1_no - 1
    return correlation_no


def parse_square_ex_index(index, step=1):
    if not isinstance(index, int):
        index = int(index)
    date_seq_no = (index // 40000) * step
    index_rem = index % 40000
    fund1_no = index_rem // 200
    fund2_no = index_rem % 200
    correlation_no = calculate_correlation_no(fund1_no, fund2_no)
    return date_seq_no, fund1_no, fund2_no, correlation_no
