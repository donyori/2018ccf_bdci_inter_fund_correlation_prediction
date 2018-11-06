from tensorflow.keras import backend as K
from tensorflow.keras import metrics

from constants import EPSILON
from data.preprocess import min_max_map, MIN_MAX_KEY_CORRELATION, MIN_MAX_KEY_TARGET


def true_mean_absolute_error(y_true, y_pred):
    yt = _restore_from_min_max_normalize(y_true)
    yp = _restore_from_min_max_normalize(y_pred)
    result = metrics.mae(y_true=yt, y_pred=yp)
    return result


def true_mean_absolute_percentage_error(y_true, y_pred):
    yt = _restore_from_min_max_normalize(y_true)
    yp = _restore_from_min_max_normalize(y_pred)
    result = metrics.mape(y_true=yt, y_pred=yp)
    return result


def targeted_mean_absolute_percentage_error(y_true, y_pred):
    result = K.mean(K.abs((y_pred - y_true) / K.clip(
        (1.5 - y_true), min_value=K.epsilon(), max_value=None)), axis=-1) * 100.
    return result


def true_targeted_mean_absolute_percentage_error(y_true, y_pred):
    yt = _restore_from_min_max_normalize(y_true)
    yp = _restore_from_min_max_normalize(y_pred)
    result = targeted_mean_absolute_percentage_error(y_true=yt, y_pred=yp)
    return result


def score(y_true, y_pred):
    mae_value = metrics.mae(y_true=y_true, y_pred=y_pred)
    tmape_value = tmape(y_true=y_true, y_pred=y_pred) / 100.
    result = K.square(2. / (2. + mae_value + tmape_value))
    return result


def true_score(y_true, y_pred):
    yt = _restore_from_min_max_normalize(y_true)
    yp = _restore_from_min_max_normalize(y_pred)
    result = score(y_true=yt, y_pred=yp)
    return result


# Alias.
tmape = TMAPE = targeted_mean_absolute_percentage_error
SCORE = score
t_mae = T_MAE = true_mean_absolute_error
t_mape = T_MAPE = true_mean_absolute_percentage_error
t_tmape = T_TMAPE = true_targeted_mean_absolute_percentage_error
t_score = T_SCORE = true_score

custom_metrics = {
    'targeted_mean_absolute_percentage_error': tmape,
    'score': score,
    'true_mean_absolute_error': t_mae,
    'true_mean_absolute_percentage_error': t_mape,
    'true_targeted_mean_absolute_percentage_error': t_tmape,
    'true_score': t_score,
}


def _restore_from_min_max_normalize(x):
    min = min_max_map[MIN_MAX_KEY_TARGET][0]
    max = min_max_map[MIN_MAX_KEY_TARGET][1]
    target_min = min_max_map[MIN_MAX_KEY_CORRELATION][0]
    target_max = min_max_map[MIN_MAX_KEY_CORRELATION][1]
    max_sub_min = max - min
    if max_sub_min < EPSILON:
        max_sub_min = EPSILON
    factor = (target_max - target_min) / max_sub_min
    y = (x - min) * factor + target_min
    return y
