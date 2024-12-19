from flask import Flask, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return 'No audio file found', 400

    audio_file = request.files['audio']
    audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename))
    return 'Audio saved successfully', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
