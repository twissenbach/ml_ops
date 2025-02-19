import random
import uuid

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class BaseModel(DeclarativeBase):
    
    @staticmethod
    def get_id():
        # If we used a random integer what would we have to do?
        # Is the hex unique, 99.99999999999% yes
        return uuid.uuid4().hex
    

db = SQLAlchemy(model_class=BaseModel)
    

# Create a table if it doesn't exist
def init_db(app, drop_db=False) -> None:
    import model_serving.models.users
    
    db.init_app(app)

    # This will re-create the table everytime it is called during the app_factory. So that we don't have to always
    # Repopulate our database, I am leaving it commented out.
    # Create the database and the User table

    if drop_db:
        with app.app_context():
            try:
                db.drop_all()
            except:
                pass

            db.create_all()
            db.session.commit()