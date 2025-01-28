import os
from dotenv import load_dotenv

load_dotenv()

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


import logging

from flask import Flask, request, jsonify

from model_serving.app_factory import create_app
from model_serving.config import Config

config = Config()


app = create_app(config)

if __name__ == '__main__':
    app.run(debug=False)