import os
from unittest import TestCase

from model_serving.config import environments
from model_serving.app_factory import create_unittest_app


class BaseTestCase(TestCase):

    os.environ['FLASK_ENV'] = 'unittest'
    config = environments[os.environ['FLASK_ENV']]

    def create_app(self):
        app = create_unittest_app(config=self.config)

        return app