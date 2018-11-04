from data.preprocess import preprocess_data, DATASET_NAME_TRAIN, DATASET_NAME_TEST, DATASET_NAME_PREDICT


def _main():
    print('Preprocess train data.')
    preprocess_data(DATASET_NAME_TRAIN)
    print('Preprocess test data.')
    preprocess_data(DATASET_NAME_TEST)
    print('Preprocess predict data.')
    preprocess_data(DATASET_NAME_PREDICT)
    print('Done.')


if __name__ == '__main__':
    _main()
