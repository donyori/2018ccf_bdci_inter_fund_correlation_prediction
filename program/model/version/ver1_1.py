from tensorflow import keras

from constants import TRADING_DAYS_PER_WEEK, INDEX_RETURN_INDICATOR_NUMBER
from ..constants import *

MODEL_NAME = 'ifcp_model_ver1_1'
ROLLING_WINDOW_SIZE = TRADING_DAYS_PER_WEEK


def build_model():
    fund1_return = keras.Input(shape=(ROLLING_WINDOW_SIZE, 1), name=FUND1_RETURN_NAME)
    fund1_benchmark_return = keras.Input(shape=(ROLLING_WINDOW_SIZE, 1), name=FUND1_BENCHMARK_RETURN_NAME)
    fund2_return = keras.Input(shape=(ROLLING_WINDOW_SIZE, 1), name=FUND2_RETURN_NAME)
    fund2_benchmark_return = keras.Input(shape=(ROLLING_WINDOW_SIZE, 1), name=FUND2_BENCHMARK_RETURN_NAME)

    fund1_performance = keras.layers.subtract([fund1_return, fund1_benchmark_return], name='fund1_performance')
    fund2_performance = keras.layers.subtract([fund2_return, fund2_benchmark_return], name='fund2_performance')

    fund1_attributes = keras.layers.concatenate(
        [fund1_return, fund1_benchmark_return, fund1_performance], name='fund1_attributes')
    fund2_attributes = keras.layers.concatenate(
        [fund2_return, fund2_benchmark_return, fund2_performance], name='fund2_attributes')

    fund_attributes_gru = keras.layers.GRU(3, name='fund_attributes_gru')

    fund1_attributes_after_gru = fund_attributes_gru(fund1_attributes)
    fund2_attributes_after_gru = fund_attributes_gru(fund2_attributes)

    fund_attributes_after_gru = keras.layers.concatenate(
        [fund1_attributes_after_gru, fund2_attributes_after_gru], name='fund_attributes_after_gru')

    auxiliary_output = keras.layers.Dense(1, activation='sigmoid', name=AUXILIARY_OUTPUT_NAME)(
        fund_attributes_after_gru)

    index_return = keras.Input(shape=(ROLLING_WINDOW_SIZE, INDEX_RETURN_INDICATOR_NUMBER), name=INDEX_RETURN_NAME)
    index_return_gru = keras.layers.GRU(35, name='index_return_gru')
    index_return_after_gru = index_return_gru(index_return)
    merge = keras.layers.concatenate([fund_attributes_after_gru, index_return_after_gru], name='merge')
    x = keras.layers.Dense(64, activation='relu')(merge)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    main_output = keras.layers.Dense(1, activation='sigmoid', name=MAIN_OUTPUT_NAME)(x)

    model = keras.Model(inputs=[
        fund1_return, fund1_benchmark_return, fund2_return, fund2_benchmark_return, index_return],
        outputs=[main_output, auxiliary_output])
    return model
