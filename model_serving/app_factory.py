# allows us to switch between environments faster

import os
import tempfile

os.environ['prometheus_multiproc_dir'] = tempfile.mkdtemp()

from flask import Flask

from model_serving.services.database.database_client import init_db
from model_serving.monitoring import init_metrics

def create_app(config):

    app = Flask(__name__)

    # Configure the SQLite database
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    app.config.from_object(config)

    init_db(app=app)
    # create_logger(app=app)
    init_blueprints(app=app)
    init_metrics(app=app)

    return app

def init_blueprints(app):
    from model_serving.routes.users import users
    app.register_blueprint(users)

    from model_serving.routes.metrics import metrics
    app.register_blueprint(metrics)

    # from model_serving.routes.prediction import prediction
    # app.register_blueprint(prediction)