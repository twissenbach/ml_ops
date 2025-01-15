import logging
import uuid

from model_serving.services.database.database_client import db
from model_serving.models.users import User

logger = logging.getLogger(__name__)

class UserController:

    def create_user(self, data):
        
        new_user = User(id=uuid.uuid4().hex, username=data['username'], email=data['email']) # if you use uuid you don't need to check for the primary key because there are so many options

        logger.debug(f'Created user id {new_user.id} for {new_user.email}')
    
        db.session.add(new_user)
        db.session.commit()

        return new_user


user_controller: UserController = UserController()