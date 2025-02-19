import logging
import time
import uuid

from model_serving.models.users import User
from model_serving.services.database.database_client import db

from sqlalchemy import delete


logger = logging.getLogger(__name__)


class UserController:

    def create_user(self, data):

        # Validate the email
        new_user = User(id=uuid.uuid4().hex, username=data['username'], email=data['email'])

        db.session.add(new_user)
        db.session.commit()

        logger.debug(f'Created user id {new_user.id} for {new_user.email}')

        return new_user
    
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
    
    def delete_user(self, user_id) -> None:

        addresses = db.session.query(User).filter(User.id == user_id)
        addresses.delete(synchronize_session=False)

        db.session.commit()

        return


user_controller: UserController = UserController()
