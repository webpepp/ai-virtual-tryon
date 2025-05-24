from flask import Flask, request, send_file
from flask_cors import CORS
from test import run_tryon

app = Flask(__name__)
CORS(app)

@app.route('/tryon', methods=['POST'])
def tryon():
    person = request.files['image']
    cloth = request.files['cloth']

    person.save('input/person.jpg')
    cloth.save('input/cloth.jpg')

    result_path = run_tryon('input/person.jpg', 'input/cloth.jpg')
    return send_file(result_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000)
