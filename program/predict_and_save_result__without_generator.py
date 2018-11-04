# Deprecated!

import csv
import os

import numpy as np

from constants import PROJECT_HOME, INDEX_RETURN_INDICATOR_NUMBER
from data.combine_data import COMBINATION_COLUMN_RANGE_KEY_FUND_RETURN, \
    COMBINATION_COLUMN_RANGE_KEY_FUND_BENCHMARK_RETURN, COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN, \
    combination_column_range_map
from data.load_dataset import DATASET_NAME_PREDICT, load_preprocessed_dataset_np
from data.preprocess import restore_correlation_from_min_max_normalize
from model.constants import FUND1_RETURN_NAME, FUND1_BENCHMARK_RETURN_NAME, \
    FUND2_RETURN_NAME, FUND2_BENCHMARK_RETURN_NAME, INDEX_RETURN_NAME
from model.metric import custom_metrics
from model.save_and_load import load_latest_model
from model.version.navi import get_latest_version_model_rolling_window_size


def _main():
    rolling_window_size = get_latest_version_model_rolling_window_size()
    dataset = load_preprocessed_dataset_np(DATASET_NAME_PREDICT)
    ds = dataset[-rolling_window_size:]
    model = load_latest_model(custom_objects=custom_metrics, does_compile=True)
    frs = combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_FUND_RETURN][0] - 1
    fbrs = combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_FUND_BENCHMARK_RETURN][0] - 1
    irr = tuple(x - 1 for x in combination_column_range_map[COMBINATION_COLUMN_RANGE_KEY_INDEX_RETURN])
    with open(os.path.join(PROJECT_HOME, 'submit.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'value'])
        for fund1_no in range(199):
            for fund2_no in range(fund1_no+1, 200):
                f1r = np.reshape(ds[:, frs+fund1_no], newshape=(1, rolling_window_size, 1))
                f1br = np.reshape(ds[:, fbrs+fund1_no], newshape=(1, rolling_window_size, 1))
                f2r = np.reshape(ds[:, frs+fund2_no], newshape=(1, rolling_window_size, 1))
                f2br = np.reshape(ds[:, fbrs+fund2_no], newshape=(1, rolling_window_size, 1))
                ir = np.reshape(
                    ds[:, irr[0]:irr[1]],
                    newshape=(1, rolling_window_size, INDEX_RETURN_INDICATOR_NUMBER)
                )
                x = {
                    FUND1_RETURN_NAME: f1r,
                    FUND1_BENCHMARK_RETURN_NAME: f1br,
                    FUND2_RETURN_NAME: f2r,
                    FUND2_BENCHMARK_RETURN_NAME: f2br,
                    INDEX_RETURN_NAME: ir,
                }
                result = model.predict(x, batch_size=1, verbose=0)[0]
                result = restore_correlation_from_min_max_normalize(result, does_overwrite=True)
                writer.writerow(['Fund %d-Fund %d' % (fund1_no+1, fund2_no+1), result[0][0]])


if __name__ == '__main__':
    _main()
