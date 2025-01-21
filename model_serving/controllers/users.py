import logging
import uuid

from model_serving.services.database.database_client import db
from model_serving.models.users import User

from sqlalchemy import delete

logger = logging.getLogger(__name__)

class UserController:

    def create_user(self, data):

        # validate the email (question 9) we figure this out how we want
        
        new_user = User(id=uuid.uuid4().hex, username=data['username'], email=data['email']) # if you use uuid you don't need to check for the primary key because there are so many options

        logger.debug(f'Created user id {new_user.id} for {new_user.email}')
    
        db.session.add(new_user)
        db.session.commit()

        return new_user
    
    def get_user(self, user_id):

        return db.session.query(User).filter(User.id == user_id).first()
    
    def modify_user(self, user_id, data: dict) -> User:

        # reuse code to get the user
        user = self.get_user(user_id)

        # if the user does not exist, error
        if not user:
            raise ValueError()
        
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

user_controller: UserController = UserController()