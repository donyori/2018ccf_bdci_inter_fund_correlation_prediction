from data.combine_data import combine_data


def _main():
    print('Combine train dataset.')
    combine_data(is_train=True)
    print('Combine test dataset.')
    combine_data(is_train=False)
    print('Done.')


if __name__ == '__main__':
    _main()
