import unittest

from model_serving.controllers.users import user_controller

class TestUser(unittest.TestCase):
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