from .path import TRADING_DATE_FILE_PATH


def number_date():
    with open(TRADING_DATE_FILE_PATH) as file:
        dates = [x.strip() for x in file.readlines()]
        date_map = dict(zip(dates, range(len(dates))))
        return date_map
