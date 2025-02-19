import os
import shutil
from unittest import TestCase
import tempfile

from model_serving.config import environments
from model_serving.app_factory import create_unittest_app


def create_metrics_dir():
    if not os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
        os.environ['PROMETHEUS_MULTIPROC_DIR'] = tempfile.mkdtemp()


class BaseTestCase(TestCase):

    os.environ['FLASK_ENV'] = 'unittest'
    config = environments[os.environ['FLASK_ENV']]

    def setUp(self):
        ...

    def create_app(self):
        app = create_unittest_app(config=self.config)

        return app