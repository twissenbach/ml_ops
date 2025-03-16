from unittest.mock import patch
import json
from ..base import BaseTestCase
from model_serving.models.users import User

class TestUserRoutes(BaseTestCase):
    @patch('model_serving.gateways.mlflow_gateway.MLFlowGateway._load_model')
    def setUp(self, mock_load_model):
        # Mock MLflow model loading
        mock_load_model.return_value = None
        super().setUp()
        self.client = self.app.test_client()
        self.test_user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass123'
        }

    def test_add_user_success(self):
        response = self.client.post(
            '/users',
            json=self.test_user_data
        )
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertIn('id', data)
        self.assertEqual(data['user_name'], 'testuser')
        self.assertEqual(data['email'], 'test@example.com')

    def test_add_user_invalid_data(self):
        # Missing required field
        invalid_data = {
            'email': 'test@example.com',
            'password': 'testpass123'
        }
        
        response = self.client.post(
            '/users',
            json=invalid_data
        )
        
        self.assertEqual(response.status_code, 406)
        self.assertIn('Invalid request data', response.json['message'])

    def test_add_user_invalid_email(self):
        invalid_data = {
            'username': 'testuser',
            'email': 'invalid_email',
            'password': 'testpass123'
        }
        
        response = self.client.post(
            '/users',
            json=invalid_data
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid email', response.json['message'].lower())

    def test_get_user_success(self):
        # First create a user
        user = User(username='testuser', email='test@example.com')
        self.db.session.add(user)
        self.db.session.commit()
        
        response = self.client.get(f'/users/{user.id}')
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertEqual(data['user_name'], 'testuser')
        self.assertEqual(data['email'], 'test@example.com')

    def test_get_user_not_found(self):
        response = self.client.get('/users/999')
        self.assertEqual(response.status_code, 404)

    def test_modify_user_success(self):
        # First create a user
        user = User(username='testuser', email='test@example.com')
        self.db.session.add(user)
        self.db.session.commit()
        
        update_data = {
            'username': 'updateduser',
            'email': 'updated@example.com'
        }
        
        response = self.client.patch(
            f'/users/{user.id}',
            json=update_data
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['user']['user_name'], 'updateduser')
        self.assertEqual(data['user']['email'], 'updated@example.com')

    def test_modify_user_not_found(self):
        update_data = {
            'username': 'updateduser',
            'email': 'updated@example.com'
        }
        
        response = self.client.patch(
            '/users/999',
            json=update_data
        )
        
        self.assertEqual(response.status_code, 404)
        self.assertIn('User does not exist', response.json['message'])

    def test_modify_user_invalid_data(self):
        # First create a user
        user = User(username='testuser', email='test@example.com')
        self.db.session.add(user)
        self.db.session.commit()
        
        invalid_data = {
            'email': 'invalid_email'
        }
        
        response = self.client.patch(
            f'/users/{user.id}',
            json=invalid_data
        )
        
        self.assertEqual(response.status_code, 400)

    def test_delete_user_success(self):
        # First create a user
        user = User(username='testuser', email='test@example.com')
        self.db.session.add(user)
        self.db.session.commit()
        
        response = self.client.delete(f'/users/{user.id}')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('User deleted successfully', response.json['message'])

    def test_delete_user_not_found(self):
        response = self.client.delete('/users/999')
        self.assertEqual(response.status_code, 404)

    def test_get_all_users_success(self):
        # Create multiple users
        users = [
            User(username='user1', email='user1@example.com'),
            User(username='user2', email='user2@example.com')
        ]
        for user in users:
            self.db.session.add(user)
        self.db.session.commit()
        
        response = self.client.get('/users')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['user_name'], 'user1')
        self.assertEqual(data[1]['user_name'], 'user2')

    def test_get_all_users_empty(self):
        response = self.client.get('/users')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 0)
