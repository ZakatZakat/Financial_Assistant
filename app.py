from flask import Flask, render_template, request, jsonify
import os
from textblob import TextBlob

app = Flask(__name__)

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    data = request.get_json()
    text = data['text']
    sentiment = analyze_sentiment(text)
    print(text)
    print(sentiment)

    log_file_path = os.path.join('saved_log', 'log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write(text + '\n')
    
    return jsonify(status="success")

@app.route('/api/analyse-log', methods=['GET'])
def analyse_log():
    log_file_path = os.path.join('saved_log', 'log.txt')
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()
    
    
if __name__ == '__main__':
    app.run(debug=True)
