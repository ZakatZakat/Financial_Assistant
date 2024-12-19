from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from Pipelines import preprocess, emotion_pipeline, translation_pipeline_rus_eng, translation_pipeline_eng_rus, summarization_pipeline, emotions_pipeline
import json
from Metrics import sentiment_metrics

app = Flask(__name__)

@app.context_processor
def inject_static_url():
    return dict(static_url=app.static_url_path)

@app.route('/')
def index():
    return render_template('in_zapis.html')


@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    data = request.get_json()
    print(data)
    text = data['text']
    log_file_path = os.path.join('saved_log', 'log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write(text + '\n')
    
    return jsonify(status="success")

@app.route('/api/analyse-log', methods=['POST'])
def analyse_log():
    label_mapping = {
    'annoyance': 'раздражение',
    "admiration": "восхищение",
    "amusement": "веселье",
    "anger": "гнев",
    "bother": "досада",
    "approval": "одобрение",
    "caring": "забота",
    "confusion": "замешательство",
    "curiosity": "любопытство",
    "desire": "желание",
    "disapproval": "неодобрение",
    "disappointment": "разочарование",
    "disgust": "отвращение",
    "embarrassment": "смущение",
    "excitement": "восторг",
    "fear": "страх",
    "gratitude": "благодарность",
    "grief": "горе",
    "joy": "радость",
    "love": "любовь",
    "nervousness": "тревога",
    "optimism": "оптимизм",
    "pride": "гордость",
    "realization": "осознание",
    "relief": "облегчение",
    "remorse": "раскаяние",
    "sadness": "грусть",
    "surprise": "удивление",
    "neutral": "нейтралитет"
    }
    
    
    result = preprocess()
    translated_text_eng_rus = translation_pipeline_rus_eng(result)
    print(translated_text_eng_rus)
    emotion_pars = emotions_pipeline(translated_text_eng_rus, label_mapping)
    print(emotion_pars)
    #emotion_scores = emotion_pipeline(emotion_pars)
    

    response = {
        'sentiment_score': emotion_pars,
        #'emotions_scores': emotion_scores
    }

    return jsonify(response)

@app.route('/api/summarize', methods=['POST'])
def summarizate_log():
    data = request.get_json()
    command = data['command']
    print(data)
    result = preprocess()
    #translated_text_eng_rus = translation_pipeline_rus_eng(result)
    summarized_text = summarization_pipeline(result, command)
    #translated_text_rus_eng = translation_pipeline_eng_rus(summarized_text)
    return jsonify(summarized_text)

if __name__ == '__main__':
    app.run(debug=True)
