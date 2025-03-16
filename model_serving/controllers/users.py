import logging
import time
import uuid

from model_serving.models.users import User
from model_serving.services.database.database_client import db

from sqlalchemy import delete


logger = logging.getLogger(__name__)


class UserController:

    def create_user(self, user_data: dict):
        # Check for existing email
        existing_user = db.session.query(User).filter(User.email == user_data['email']).first()
        if existing_user:
            raise ValueError("Email already exists")
        
        user = User(
            id=uuid.uuid4().hex,
            username=user_data['username'],
            email=user_data['email']
        )
        db.session.add(user)
        db.session.commit()
        return user
    
    def get_user(self, user_id):

        return db.session.query(User).filter(User.id == user_id).first()

    def modify_user(self, user_id: str, data: dict) -> User:

        # Reuse code to get the user
        user = self.get_user(user_id)

        # If the user does not exist, error
        if not user:
            raise ValueError()
        
        # Update if it does exist
        if 'email' in data:
            user.email = data['email']

        if 'user_name' in data:
            user.username = data['user_name']

        # Save
        db.session.merge(user)
        db.session.commit()

        # Return
        return user
    
    def delete_user(self, user_id: str):
        user = db.session.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")
        db.session.delete(user)
        db.session.commit()

    def get_users(self):
        """Get all users"""
        return db.session.query(User).all()


user_controller: UserController = UserController()
