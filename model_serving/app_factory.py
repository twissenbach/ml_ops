import os
import tempfile

os.environ['prometheus_multiproc_dir'] = tempfile.mkdtemp()

from flask import Flask

from model_serving.services.database.database_client import init_db
from model_serving.services.inference_service import inference_service
from model_serving.gateways.mlflow_gateway import mlflow_gateway
from model_serving.monitoring import init_metrics


def create_app(config):
    
    app = Flask(__name__)

    # Configure the SQLite database
    app.config.from_object(config)

    # Configure the SQLite database
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    init_db(app=app)
    init_blueprints(app=app)
    init_metrics(app=app)
    # inference_service(app=app)
    mlflow_gateway.init_app(app=app)
    
    return app


def create_unittest_app(config):

    app = Flask(__name__)

    app.config.from_object(config)

    # Configure the SQLite database
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    init_db(app=app, drop_db=True)
    init_blueprints(app=app)
    init_metrics(app=app)
    # inference_service(app=app)
    mlflow_gateway.init_app(app=app)

    return app


def init_blueprints(app):
    from model_serving.routes.users import users
    app.register_blueprint(users)
    
    from model_serving.routes.metrics import metrics
    app.register_blueprint(metrics)
    
    from model_serving.routes.prediction import prediction
    app.register_blueprint(prediction)