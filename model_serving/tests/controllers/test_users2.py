import unittest
import sys
import os
from dotenv import load_dotenv

print("prometheus multiproc dir")
print(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))

# Add this debug print
print("Python Path:", sys.path)
print("Current Directory:", os.getcwd())

load_dotenv()  # Load environment variables from .env file

MY_VARIABLE = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
print("prometheus multiproc dir:", MY_VARIABLE)

# Modify the path append to use absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from model_serving.controllers.users import user_controller

class TestUser(unittest.TestCase):
    

    def setUp(self):
        ...

    def test_user_controller_create(self, args, **kwargs):
        # setup input
        data = {
            'username': 'trwissen',
            'email': 'trwissen@gmail.com'
        }

        # run the function
        response = user_controller.create_user(data)

        self.assertEqual(len(response), 3)
        