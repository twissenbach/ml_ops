import re
import logging
import time
import uuid

from model_serving.services.database.database_client import db
from model_serving.models.users import User
# from model_serving.monitoring import DATABASE_LATENCY

from sqlalchemy import delete

logger = logging.getLogger(__name__)

class UserController:

    def __init__(self):
        # Simple email regex pattern
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    def validate_email(self, email: str) -> bool:
        return bool(self.email_pattern.match(email))

    def create_user(self, data):
        # Validate email format
        if not self.validate_email(data['email']):
            raise ValueError("Invalid email format")

        # Check if email already exists
        existing_user = db.session.query(User).filter(User.email == data['email']).first()
        if existing_user:
            raise ValueError("Email already exists")
        
        new_user = User(id=uuid.uuid4().hex, username=data['username'], email=data['email'])
        logger.debug(f'Created user id {new_user.id} for {new_user.email}')
        
        db.session.add(new_user)
        db.session.commit()

        return new_user
    
    def get_user(self, user_id):

        return db.session.query(User).filter(User.id == user_id).first()
    
    def modify_user(self, user_id, data: dict) -> User:
        # Check if required fields are present
        required_fields = ['email', 'user_name']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")

        # reuse code to get the user
        user = self.get_user(user_id)

        # if the user does not exist, error
        if not user:
            raise ValueError("User not found")
        
        # update if it does exist
        user.email = data['email']
        user.username = data['user_name']

        # save
        db.session.merge(user)
        db.session.commit()

        # return
        return user
    
    def delete_user(self, user_id) -> None:

        stmt = delete(User).where(User.id == user_id)
        db.session.query(stmt)

        db.session.commit()

        return

    def get_users(self):
        """Get all users from the database"""
        return db.session.query(User).all()

user_controller: UserController = UserController()