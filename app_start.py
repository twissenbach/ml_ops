# curl -X POST http://127.0.0.1:5000/users -H "Content-Type: application/json" -d '{"username": "johndoe", "email": "johndoe@example.com"}'
# curl -X GET http://127.0.0.1:5000/users

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define a User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# Create the database and the User table
with app.app_context():
    db.create_all()

# function
# @app.route('/users', methods=['POST']) # Route
# def add_user():
#     data = request.json
#     new_user = User(username=data['username'], email=data['email'])
#     db.session.add(new_user)
#     db.session.commit()
#     return jsonify({'message': 'User created successfully!'}), 201

@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    output = []
    for user in users:
        output.append({'id': user.id, 'username': user.username, 'email': user.email})
    return jsonify(output), 200

if __name__ == '__main__':
    app.run(debug=True)
