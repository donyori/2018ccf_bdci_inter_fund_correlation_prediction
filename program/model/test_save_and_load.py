import unittest

from .compile import compile_model
from .constants import MAIN_OUTPUT_NAME, AUXILIARY_OUTPUT_NAME
from .metric import custom_metrics
from .save_and_load import save_model, load_model
from .version.navi import get_latest_version_model_name, build_latest_version_model


class TestSaveAndLoad(unittest.TestCase):

    def test_save_model(self):
        model = build_latest_version_model()
        print('build finish.')
        model.compile(optimizer='rmsprop', loss='mse', loss_weights={MAIN_OUTPUT_NAME: 1., AUXILIARY_OUTPUT_NAME: 0.2})
        print('compile finish.')
        save_model(get_latest_version_model_name(), model, does_include_optimizer=True)
        print('save finish.')

    def test_save_and_load_model(self):
        model = build_latest_version_model()
        print('build finish.')
        compile_model(model)
        print('compile finish.')
        model_name = get_latest_version_model_name()
        save_model(model_name, model, does_include_optimizer=True)
        print('save finish.')
        # This should fail:
        try:
            load_model(model_name, does_compile=True)
        except Exception as e:
            self.assertNotEqual(e, None)
        # This should succeed:
        loaded_model = load_model(model_name, custom_objects=custom_metrics, does_compile=True)
        print('load finish.')
        self.assertNotEqual(loaded_model, None)
