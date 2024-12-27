from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST']) 
def home_page():
    return "<h1>Movie recommendation api</h1>"

if __name__ == '__main__':
    app.run()  