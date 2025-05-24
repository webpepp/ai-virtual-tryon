from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from test import run_tryon  # assume your test logic is in test.py

app = Flask(__name__)
CORS(app)

@app.route('/tryon', methods=['POST'])
def tryon():
    image = request.files['image']
    cloth = request.files['cloth']

    image.save('input/person.jpg')
    cloth.save('input/cloth.jpg')

    output_path = run_tryon('input/person.jpg', 'input/cloth.jpg')

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000)
