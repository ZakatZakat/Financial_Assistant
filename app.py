from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Маршрут для отдачи HTML страницы
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    data = request.get_json()
    print(data['text'])
    #print("Received text:", text)  # Здесь можно добавить обработку текста
    return jsonify(status="success")

if __name__ == '__main__':
    app.run(debug=True)
