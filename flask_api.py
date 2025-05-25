from flask import Flask, request, send_file
from flask_cors import CORS
import os
from test import run_tryon  # assumes test.py contains run_tryon()

app = Flask(__name__)

# ✅ Fix: Allow preflight OPTIONS and credentials (if needed)
CORS(app, supports_credentials=True, resources={
    r"/tryon": {
        "origins": ["https://pepphr.com"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Ensure directories exist
os.makedirs('input', exist_ok=True)
os.makedirs('output', exist_ok=True)

@app.route('/tryon', methods=['POST'])
def tryon():
    if 'image' not in request.files or 'cloth' not in request.files:
        return 'Missing image or cloth file', 400

    person = request.files['image']
    cloth = request.files['cloth']

    person_path = 'input/person.jpg'
    cloth_path = 'input/cloth.jpg'

    person.save(person_path)
    cloth.save(cloth_path)

    try:
        result_path = run_tryon(person_path, cloth_path)
        return send_file(result_path, mimetype='image/png')
    except Exception as e:
        print("❌ Error in /tryon:", str(e))  # This shows in Colab
        return f"Internal Server Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
