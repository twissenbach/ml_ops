from pathlib import Path
import os


class Config:
    ENV = os.environ.get('FLACK_ENV', 'development')
    APP_NAME = 'user_application'
    WORKING_DIRECTORY = str(Path(__file__).parent / 'user_application')

    # logger
    LOG_LEVEL = 'INFO'

    # SQL Alchemy
    SQLALCHEMY_DATABASE_URI = 'sqlite:///users.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class UnitTest(Config):
    SQLALCHEMY_DATABASE_URL = 'sqlite:///:memory:'

    