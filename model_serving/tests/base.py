import os
import shutil
from unittest import TestCase
import tempfile
import unittest
from model_serving.app_factory import create_app
from model_serving.services.database.database_client import db
from model_serving.tests.config import TestConfig

from model_serving.config import environments
from model_serving.app_factory import create_unittest_app


def create_metrics_dir():
    metrics_dir = tempfile.mkdtemp()
    os.environ['prometheus_multiproc_dir'] = metrics_dir
    return metrics_dir


class BaseTestCase(unittest.TestCase):

    os.environ['FLASK_ENV'] = 'unittest'
    config = environments[os.environ['FLASK_ENV']]

    def setUp(self):
        self.app = create_app(TestConfig)
        self.client = self.app.test_client()
        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def create_app(self):
        app = create_unittest_app(config=self.config)

        return app