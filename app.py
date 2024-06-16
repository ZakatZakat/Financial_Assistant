from flask import Flask, render_template, request, jsonify
import os
from Pipelines import preprocess, emotion_pipeline, translation_pipeline_rus_eng, translation_pipeline_eng_rus, summarization_pipeline
import json
from Metrics import sentiment_metrics

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    data = request.get_json()
    text = data['text']
    log_file_path = os.path.join('saved_log', 'log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write(text + '\n')
    
    return jsonify(status="success")

@app.route('/api/analyse-log', methods=['POST'])
def analyse_log():
    result, label_mapping = preprocess()
    translated_text = translation_pipeline_rus_eng(result)
    sentiment_score = sentiment_metrics(translated_text)
    emotions_scores = emotion_pipeline(translated_text, label_mapping)

    response = {
        'sentiment_score': sentiment_score,
        'emotions_scores': emotions_scores
    }

    return jsonify(response)

@app.route('/api/summarize', methods=['POST'])
def summarizate_log():
    result, _ = preprocess()
    translated_text_eng_rus = translation_pipeline_rus_eng(result)
    summarized_text = summarization_pipeline(translated_text_eng_rus)
    translated_text_rus_eng = translation_pipeline_eng_rus(summarized_text)
    return jsonify(translated_text_rus_eng)

if __name__ == '__main__':
    app.run(debug=True)
