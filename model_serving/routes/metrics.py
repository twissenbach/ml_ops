from flask import Blueprint, jsonify, request
from prometheus_client import generate_latest

from model_serving.monitoring import registry


metrics = Blueprint('metrics', __name__)


@metrics.route('/metrics', methods=['GET'])
def get_metrics():
    return generate_latest(registry), 200, {'Content-Type': 'test/plain: version=0.0.4: charset=utf-8'}