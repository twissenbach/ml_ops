# users.py route

from flask import Blueprint, jsonify, request
import logging

from model_serving.models.users import User
from model_serving.services.database.database_client import db
from model_serving.controllers.users import user_controller


users = Blueprint('users', __name__)
logger = logging.getLogger(__name__) # You technically don't need this line. You could call the library directly. This line pulls extra information.


@users.route('/users', methods=['POST']) # Route
def add_user():

    data = request.json

    try:
        data = user_controller.create_user(data)
    except KeyError as e:
        logger.exception(f'Encountered error {str(e)}')
        return jsonify({'message': 'User created successfully!'}), 406
    except Exception as e:
        logger.exception(f'Encountered Error {str(e)}')
        return jsonify({'message': 'User created successfully!'}), 404

    return jsonify({'id': data.id, 'user_name': data.username, 'email': data.email}), 201

@users.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):

    data = user_controller.get_user(user_id)
    return jsonify({'id': data.id, 'user_name': data.username, 'email': data.email}), 201

@users.route('/users/<user_id>', methods=['PATCH'])
def modify_user(user_id):

    data = request.json

    try:
        user = user_controller.modify_user(user_id, data)
    except ValueError as e:
        logger.exception(f'Encountered error {str(e)}')
        return jsonify({'id': user_id, 'message': 'User does not exist'}), 404

    return jsonify({'id': user.id, 'user_name': user.username, 'email': user.email}), 201

@users.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):

    user_controller.delete_user(user_id)

    return jsonify({'id': user_id, 'message': 'User deleted successfully'}), 200

@users.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    output = []
    for user in users:
        output.append({'id': user.id, 'user_name': user.username, 'email': user.email})
    return jsonify(output), 200