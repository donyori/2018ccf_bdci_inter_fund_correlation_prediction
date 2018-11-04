import unittest

from .navi import get_all_valid_model_names
from .ver1_0 import MODEL_NAME as MN1_0
from .ver1_1 import MODEL_NAME as MN1_1


class TestNavi(unittest.TestCase):

    def test_get_all_valid_model_names(self):
        mns = get_all_valid_model_names()
        print(mns)
        self.assertEqual(mns, [MN1_0, MN1_1])
