import numpy as np

from .combine_data import combination_column_range_map
from .load_dataset import load_dataset_np


def statistics(dataset):
    if isinstance(dataset, str):
        dataset = load_dataset_np(dataset_name=dataset)
    if not isinstance(dataset, np.ndarray):
        raise TypeError('dataset must be np.ndarray or the name of dataset(string).')
    min = np.nanmin(dataset)
    max = np.nanmax(dataset)
    median = np.nanmedian(dataset)
    mean = np.nanmean(dataset)
    std = np.nanstd(dataset)
    var = np.nanvar(dataset)
    result = {
        'min': min,
        'max': max,
        'median': median,
        'mean': mean,
        'std': std,
        'var': var,
    }
    return result


def statistics_on_every_fields(dataset):
    if isinstance(dataset, str):
        dataset = load_dataset_np(dataset_name=dataset)
    if not isinstance(dataset, np.ndarray):
        raise TypeError('dataset must be np.ndarray or the name of dataset(string).')
    result = dict()
    inner_size = dataset.shape[-1]
    for field_name, column_range in combination_column_range_map.items():
        if column_range[1] - 1 > inner_size:
            continue
        sub_dataset = dataset[:, column_range[0]-1: column_range[1]-1]
        min = np.nanmin(sub_dataset)
        max = np.nanmax(sub_dataset)
        median = np.nanmedian(sub_dataset)
        mean = np.nanmean(sub_dataset)
        std = np.nanstd(sub_dataset)
        var = np.nanvar(sub_dataset)
        d = {
            'min': min,
            'max': max,
            'median': median,
            'mean': mean,
            'std': std,
            'var': var,
        }
        result[field_name] = d
    return result


def show_statistics(dataset):
    result = statistics(dataset)
    print('Dataset:', dataset if isinstance(dataset, str) else dataset.shape)
    print(result)


def show_statistics_on_every_fields(dataset):
    result = statistics_on_every_fields(dataset)
    print('Dataset:', dataset if isinstance(dataset, str) else dataset.shape)
    print(result)
