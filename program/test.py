import argparse

from model.test import test_model, test_latest_model


def _main():
    parser = argparse.ArgumentParser(description='Arguments for testing model.')
    parser.add_argument('-m', '--model', type=str, dest='model',
                        help='The name of the model to be trained.', default=None, required=False)
    args = parser.parse_args()
    print('Start testing.')
    if args.model is not None:
        result_map = test_model(model_name=args.model)
    else:
        result_map = test_latest_model()
    print('Done.')
    print()
    print('Result:')
    print(result_map)


if __name__ == '__main__':
    _main()
