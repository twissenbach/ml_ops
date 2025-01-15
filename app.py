import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from dotenv import load_dotenv

import logging

from flask import Flask, request, jsonify

from model_serving.app_factory import create_app


app = create_app(None)

if __name__ == '__main__':
    app.run(debug=False)