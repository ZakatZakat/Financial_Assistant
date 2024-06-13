from transformers import pipeline
import json

def preprocess():
    label_mapping = {
        "admiration": "восхищение",
        "amusement": "развлечение",
        "anger": "гнево",
        "bother": "нудьга",
        "approval": "одобрение",
        "caring": "забота",
        "confusion": "замешательство",
        "curiosity": "интерес",
        "desire": "притязание",
        "disapproval": "негативное отношение",
        "disappointment": "разочарование",
        "disgust": "отвратение",
        "embarrassment": "смущение",
        "excitement": "восторг",
        "fear": "страх",
        "gratitude": "благодарность",
        "grief": "горе",
        "joy": "радость",
        "love": "любовь",
        "nervousness": "нервозность",
        "optimism": "оптимизм",
        "pride": "гордость",
        "realization": "осознание",
        "relief": "облегчение",
        "remorse": "признак",
        "sadness": "грусть",
        "surprise": "удивление",
        "neutral": "нейтралитет"
    }

    with open('saved_log/log.txt', 'r') as file:
        rows = file.readlines()
        last_row = rows[-1].strip() if rows else None
        if last_row is None:
            print("File is empty.")

        file.close()
    return last_row, label_mapping

def sentiment_tempreture(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model='ProsusAI/finbert')

    result = sentiment_pipeline(text, top_k=3)
    return result

def translation_pipeline(text):
    translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")
    translated_text = translation_pipeline(text)[0]['translation_text']
    return translated_text

def emotion_pipeline(text, label_mapping):
    # Create a pipeline for text classification
    classification_pipeline = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")

    # Perform text classification
    results = classification_pipeline(text, top_k=5)

    # Create a dictionary to store the results in the desired format
    formatted_results = {label_mapping.get(result['label'], result['label']): result['score'] for result in results}

    # Convert the dictionary to JSON and print the output
    return formatted_results
