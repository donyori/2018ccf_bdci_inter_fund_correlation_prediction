import argparse
import csv
import os

import numpy as np

from constants import PROJECT_HOME
from data.preprocess import restore_correlation_from_min_max_normalize
from model.predict import predict_last_row, predict_last_row_using_latest_model


def _main():
    parser = argparse.ArgumentParser(description='Arguments for predicting.')
    parser.add_argument('-m', '--model', type=str, dest='model',
                        help='The name of the model to be trained.', default=None, required=False)
    args = parser.parse_args()
    print('Start predicting.')
    if args.model is not None:
        result = predict_last_row(model_name=args.model)[0]
    else:
        result = predict_last_row_using_latest_model()[0]
    print('Done.')
    result = np.reshape(result, newshape=(19900,))
    print()
    print('Result:')
    print('Before restore:')
    print('Min:', np.min(result), 'Max:', np.max(result))
    result = restore_correlation_from_min_max_normalize(result, does_overwrite=True)
    print('After restore:')
    print('Min:', np.min(result), 'Max:', np.max(result))
    with open(os.path.join(PROJECT_HOME, 'submit.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'value'])
        idx = 0
        for fund1_no in range(199):
            for fund2_no in range(fund1_no+1, 200):
                writer.writerow(['Fund %d-Fund %d' % (fund1_no+1, fund2_no+1), result[idx]])
                idx += 1


if __name__ == '__main__':
    _main()
