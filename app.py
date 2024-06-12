from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Маршрут для отдачи HTML страницы
@app.route('/')
def index():
    return render_template('index.html')

# API для обработки аудио
@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' in request.files:
        audio_file = request.files['audio']
        # Обработка файла аудио вашей ASR моделью
        text = 'Пример распознанного текста'
        return jsonify(text=text)
    return jsonify({"error": "No audio file provided"}), 400


if __name__ == '__main__':
    app.run(debug=True)
