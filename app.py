from flask import Flask, request, jsonify
#import your_asr_model  # Подразумевается, что модель ASR импортируется здесь

app = Flask(__name__)

@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    audio_file = request.files['audio']
    text = your_asr_model.recognize(audio_file)  # Обработка файла аудио вашей моделью ASR
    return jsonify(text=text)

if __name__ == '__main__':
    app.run(debug=True)
