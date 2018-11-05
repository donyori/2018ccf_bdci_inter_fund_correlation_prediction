import argparse

from model.train import resume_training_model, resume_training_latest_model


def _main():
    parser = argparse.ArgumentParser(description='Arguments for training model.')
    parser.add_argument('-m', '--model', type=str, dest='model',
                        help='The name of the model to be trained.', default=None, required=False)
    parser.add_argument('-e', '--end_epoch', type=int, dest='end_epoch',
                        help='The end epoch of train.', default=1, required=False)
    args = parser.parse_args()
    print('Start training.')
    if args.model is not None:
        resume_training_model(model_name=args.model, end_epoch=args.end_epoch)
    else:
        resume_training_latest_model(end_epoch=args.end_epoch)
    print('Done.')


if __name__ == '__main__':
    _main()
