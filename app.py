from flask import Flask, render_template, request, jsonify
import os
from test_pipe import test_pipeline
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    data = request.get_json()
    text = data['text']
    print(text)

    log_file_path = os.path.join('saved_log', 'log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write(text + '\n')
    
    return jsonify(status="success")

@app.route('/api/analyse-log', methods=['POST'])
def analyse_log():
    print("Request received for analyse_log")
    result = test_pipeline()
    return result, 200
    
if __name__ == '__main__':
    app.run(debug=True)
