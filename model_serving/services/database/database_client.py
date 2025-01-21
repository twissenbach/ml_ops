# creates our database object

import random
import uuid

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class BaseModel(DeclarativeBase):

    @staticmethod
    def get_id():
        # if we used a random integer what would we have to do?
        # is the ehx unique, 99.99999999% yes
        return uuid.uuid4().hex
    
db = SQLAlchemy(model_class=BaseModel)


# create a table if it doesn't exist
def init_db(app) -> None:
    import model_serving.models.users

    db.init_app(app)

    # # create the database and the User table
    # with app.app_context():
    #     try:
    #         db.drop_all()
    #     except:
    #         pass

    #     db.create_all()
    #     db.session.commit()