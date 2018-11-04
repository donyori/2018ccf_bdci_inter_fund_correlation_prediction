from .ver1_0 import MODEL_NAME as MN1_0, ROLLING_WINDOW_SIZE as RWS1_0, build_model as bm1_0
from .ver1_1 import MODEL_NAME as MN1_1, ROLLING_WINDOW_SIZE as RWS1_1, build_model as bm1_1


class ModelInfo(object):

    def __init__(self, model_name, version_no, rolling_window_size, build_model_fn):
        if not isinstance(model_name, str):
            raise TypeError('model_name should be str.')
        if not isinstance(version_no, str):
            raise TypeError('version_no should be str.')
        if not isinstance(rolling_window_size, int):
            rolling_window_size = int(rolling_window_size)
        if not callable(build_model_fn):
            raise TypeError('build_model_fn should be callable.')
        self.model_name = model_name
        self.version_no = version_no
        self.rolling_window_size = rolling_window_size
        self.build_model_fn = build_model_fn

    def __str__(self):
        return 'Model %s(rolling_window_size=%d)' % (self.model_name, self.rolling_window_size)

    def __repr__(self):
        return str(self)


_model_info_map = dict()
_latest_version_model_info = None


def _init():
    global _model_info_map
    global _latest_version_model_info
    info = ModelInfo(model_name=MN1_0, version_no='1.0', rolling_window_size=RWS1_0, build_model_fn=bm1_0)
    _model_info_map[info.model_name] = info
    info = ModelInfo(model_name=MN1_1, version_no='1.1', rolling_window_size=RWS1_1, build_model_fn=bm1_1)
    _model_info_map[info.model_name] = info
    _latest_version_model_info = info


_init()


def get_model_info(model_name):
    if model_name not in _model_info_map:
        raise UnknownModelNameException(model_name)
    return _model_info_map[model_name]


def get_version_no(model_name):
    info = get_model_info(model_name=model_name)
    return info.version_no


def get_rolling_window_size(model_name):
    info = get_model_info(model_name=model_name)
    return info.rolling_window_size


def get_build_model_function(model_name):
    info = get_model_info(model_name=model_name)
    return info.build_model_fn


def build_model(model_name):
    bmfn = get_build_model_function(model_name=model_name)
    model = bmfn()
    return model


def get_latest_version_model_name():
    return _latest_version_model_info.model_name


def get_latest_version_model_info():
    return _latest_version_model_info


def get_latest_version_model_version_no():
    return _latest_version_model_info.version_no


def get_latest_version_model_rolling_window_size():
    return _latest_version_model_info.rolling_window_size


def get_latest_version_model_build_model_function():
    return _latest_version_model_info.build_model_fn


def build_latest_version_model():
    bmfn = get_latest_version_model_build_model_function()
    model = bmfn()
    return model


def get_all_valid_model_names():
    mns = list()
    mns.extend(_model_info_map.keys())
    mns.sort()
    return mns


class UnknownModelNameException(ValueError):

    def __init__(self, model_name):
        super().__init__('model_name "%s" is unknown.' % model_name)
