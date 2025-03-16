import pytest
from unittest.mock import Mock, patch
import uuid
from sqlalchemy import delete
from flask import Flask
from model_serving.routes.users import users  # Import the Blueprint

from model_serving.controllers.users import UserController
from model_serving.models.users import User

@pytest.fixture
def user_controller():
    return UserController()

@pytest.fixture
def mock_db_session():
    with patch('model_serving.controllers.users.db.session') as mock_session:
        yield mock_session

@pytest.fixture
def sample_user_data():
    return {
        'username': 'testuser',
        'email': 'test@example.com'
    }

@pytest.fixture
def sample_user():
    return User(
        id=uuid.uuid4().hex,
        username='testuser',
        email='test@example.com'
    )

class TestUserController:
    def test_create_user_basic(self, user_controller, mock_db_session, sample_user_data):
        # Configure mock to return None for the email existence check
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Test basic user creation with valid data
        result = user_controller.create_user(sample_user_data)
        
        # Assert the user was created with correct data
        assert result.username == sample_user_data['username']
        assert result.email == sample_user_data['email']
        assert hasattr(result, 'id')
        
        # Verify database interactions
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_get_user_basic(self, user_controller, mock_db_session, sample_user):
        # Configure mock to return the sample user
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user
        
        # Test getting user by ID
        result = user_controller.get_user(sample_user.id)
        
        # Assert the correct user was returned
        assert result.id == sample_user.id
        assert result.username == sample_user.username
        assert result.email == sample_user.email
        
        # Verify database was queried
        mock_db_session.query.assert_called_once()

    def test_modify_user_basic(self, user_controller, mock_db_session, sample_user):
        # Configure mock to return the sample user for the existence check
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user
        
        # Test data for update
        update_data = {
            'email': 'updated@example.com',
            'user_name': 'updateduser'
        }
        
        # Test modifying user
        result = user_controller.modify_user(sample_user.id, update_data)
        
        # Assert the user was updated with correct data
        assert result.id == sample_user.id
        assert result.username == update_data['user_name']
        assert result.email == update_data['email']
        
        # Verify database interactions
        mock_db_session.merge.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_delete_user_basic(self, user_controller, mock_db_session):
        user_id = "test-user-id"
        
        # Mock the query result
        mock_user = User(id=user_id, username="test", email="test@example.com")
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_user
        
        # Test deleting user
        user_controller.delete_user(user_id)
        
        # Verify database interactions
        mock_db_session.query.assert_called_once_with(User)
        mock_db_session.delete.assert_called_once_with(mock_user)
        mock_db_session.commit.assert_called_once()

    def test_get_all_users_basic(self, user_controller, mock_db_session):
        sample_users = [
            User(id='1', username='user1', email='user1@example.com'),
            User(id='2', username='user2', email='user2@example.com')
        ]
        mock_db_session.query.return_value.all.return_value = sample_users
        
        result = user_controller.get_users()
        assert result == sample_users

    def test_create_user_duplicate_email(self, user_controller, mock_db_session):
        sample_user_data = {'email': 'test@example.com', 'username': 'testuser'}
        
        # First call - success
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        result = user_controller.create_user(sample_user_data)
        assert result.email == sample_user_data['email']
        
        # Reset the mock to simulate second call with existing email
        mock_db_session.reset_mock()
        existing_user = User(id='1', email=sample_user_data['email'], username='existing')
        mock_db_session.query.return_value.filter.return_value.first.return_value = existing_user
        
        with pytest.raises(ValueError, match="Email already exists"):
            user_controller.create_user(sample_user_data)

    def test_modify_user_success_message(self, user_controller, mock_db_session, sample_user):
        app = Flask(__name__)
        app.register_blueprint(users)
        
        # Configure mock to return the sample user for the existence check
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user
        
        # Test data for update
        update_data = {
            'email': 'updated@example.com',
            'user_name': 'updateduser'
        }
        
        with app.test_client() as client:
            with patch('model_serving.routes.users.user_controller') as mock_controller:
                mock_controller.modify_user.return_value = sample_user
                
                # Make the request
                response = client.patch(
                    f'/users/{sample_user.id}',
                    json=update_data
                )
                
                # Assert response
                assert response.status_code == 200
                response_data = response.get_json()
                assert response_data['message'] == 'User details updated successfully'
                assert 'user' in response_data
                assert response_data['user']['id'] == sample_user.id

    def test_modify_user_error_cases(self, user_controller, mock_db_session, sample_user):
        app = Flask(__name__)
        app.register_blueprint(users)
        
        # Test Case 1: Missing required fields
        incomplete_data = {
            'email': 'test@example.com'
            # missing user_name field
        }
        
        with app.test_client() as client:
            with patch('model_serving.routes.users.user_controller') as mock_controller:
                mock_controller.modify_user.side_effect = KeyError("Missing required fields: user_name")
                
                response = client.patch(
                    f'/users/{sample_user.id}',
                    json=incomplete_data
                )
                
                # Assert response for missing fields
                assert response.status_code == 400
                response_data = response.get_json()
                assert "Missing required fields" in response_data['message']
        
        # Test Case 2: User not found
        valid_data = {
            'email': 'test@example.com',
            'user_name': 'testuser'
        }
        
        with app.test_client() as client:
            with patch('model_serving.routes.users.user_controller') as mock_controller:
                mock_controller.modify_user.side_effect = ValueError("User not found")
                
                response = client.patch(
                    '/users/nonexistent-id',
                    json=valid_data
                )
                
                # Assert response for user not found
                assert response.status_code == 404
                response_data = response.get_json()
                assert response_data['message'] == 'User does not exist'

    def test_email_validation(self, user_controller, mock_db_session):
        app = Flask(__name__)
        app.register_blueprint(users)
        
        # Test Case 1: Valid email format
        valid_data = {
            'username': 'testuser',
            'email': 'valid@example.com'
        }
        
        with app.test_client() as client:
            with patch('model_serving.routes.users.user_controller') as mock_controller:
                mock_controller.create_user.return_value = User(
                    id='test-id',
                    username=valid_data['username'],
                    email=valid_data['email']
                )
                
                response = client.post('/users', json=valid_data)
                
                # Assert successful creation
                assert response.status_code == 201
                response_data = response.get_json()
                assert response_data['email'] == valid_data['email']
        
        # Test Case 2: Invalid email formats
        invalid_emails = [
            'notanemail',           # No @ symbol
            '@nodomain',            # No local part
            'no.domain@',           # No domain part
            'spaces in@email.com',  # Contains spaces
            'wrong@domain.',        # Incomplete domain
        ]
        
        with app.test_client() as client:
            with patch('model_serving.routes.users.user_controller') as mock_controller:
                mock_controller.create_user.side_effect = ValueError("Invalid email format")
                
                for invalid_email in invalid_emails:
                    invalid_data = {
                        'username': 'testuser',
                        'email': invalid_email
                    }
                    
                    response = client.post('/users', json=invalid_data)
                    
                    # Assert validation failure
                    assert response.status_code == 400
                    response_data = response.get_json()
                    assert response_data['message'] == 'Invalid email format'

