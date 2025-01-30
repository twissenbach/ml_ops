from pathlib import Path
import os


class Config:
    ENV = os.environ.get('FLASK_ENV', 'development')
    APP_NAME = 'user_application'
    # WORKING_DIRECTORY = str(Path(__file__).parent / 'user_application')
    WORKING_DIRECTORY = current_path = str(Path(__name__).resolve().parent) + '\\model_serving\\user_application'

    # logger
    # LOG_LEVEL = 'INFO'
    LOG_LEVEL = 'DEBUG'

    # SQL Alchemy
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///users.db'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///user_application.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class UnitTest(Config):
    SQLALCHEMY_DATABASE_URL = 'sqlite:///:memory:'

    
class Production(Config):
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

environments = {
    'config': Config(),
    'unit_test': UnitTest(),
}