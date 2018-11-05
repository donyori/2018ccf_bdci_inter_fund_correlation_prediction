DATASET_NAME_TRAIN = 'train'
DATASET_NAME_TEST = 'test'
DATASET_NAME_PREDICT = 'predict'


class UnknownDatasetNameException(ValueError):

    def __init__(self, dataset_name):
        super().__init__('dataset_name "%s" is unknown.' % dataset_name)
