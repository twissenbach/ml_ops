import pytest
from unittest.mock import Mock, patch
import uuid

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
    
    def test_create_user_success(self, user_controller, mock_db_session, sample_user_data):
        # Test successful user creation
        new_user = user_controller.create_user(sample_user_data)
        
        assert new_user.username == sample_user_data['username']
        assert new_user.email == sample_user_data['email']
        assert isinstance(new_user.id, str)
        
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_create_user_missing_data(self, user_controller):
        # Test user creation with missing data
        with pytest.raises(KeyError):
            user_controller.create_user({'username': 'testuser'})  # Missing email

    def test_get_user_success(self, user_controller, mock_db_session, sample_user):
        # Setup mock query
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user
        
        # Test successful user retrieval
        user = user_controller.get_user(sample_user.id)
        
        assert user == sample_user
        mock_db_session.query.assert_called_once_with(User)

    def test_get_user_not_found(self, user_controller, mock_db_session):
        # Setup mock query to return None
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Test user retrieval when user doesn't exist
        user = user_controller.get_user('nonexistent_id')
        
        assert user is None

    def test_modify_user_success(self, user_controller, mock_db_session, sample_user):
        # Setup mock get_user
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_user
        
        # Test successful user modification
        modified_data = {
            'email': 'new@example.com',
            'user_name': 'newusername'
        }
        
        modified_user = user_controller.modify_user(sample_user.id, modified_data)
        
        assert modified_user.email == modified_data['email']
        assert modified_user.username == modified_data['user_name']
        mock_db_session.merge.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_modify_user_not_found(self, user_controller, mock_db_session):
        # Setup mock get_user to return None
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(ValueError):
            user_controller.modify_user('nonexistent_id', {'email': 'new@example.com', 'user_name': 'newname'})

    def test_delete_user(self, user_controller, mock_db_session):
        # Test user deletion
        user_id = 'test_id'
        user_controller.delete_user(user_id)
        
        mock_db_session.query.assert_called_once()
        mock_db_session.commit.assert_called_once()

