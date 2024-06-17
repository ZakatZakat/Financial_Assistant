from transformers import pipeline
import json

def preprocess():
    # Словарь для маппинга английских эмоциональных меток на русские
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
    # Чтение данных из файла log.txt и извлечение ПОСЛЕДНЕЙ записи
    with open('saved_log/log.txt', 'r') as file:
        rows = file.readlines()
        last_row = rows[-1].strip() if rows else None
        if last_row is None:
            print("File is empty.")

        file.close()
    return last_row, label_mapping

def sentiment_tempreture(text):
    # Создание объекта pipeline для анализа тональности с использованием модели finbert
    sentiment_pipeline = pipeline("sentiment-analysis", model='ProsusAI/finbert')

    result = sentiment_pipeline(text, top_k=3)
    return result

def translation_pipeline_rus_eng(text):
    # Создание объекта pipeline для перевода текста с русского на английский
    translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")
    translated_text = translation_pipeline(text)[0]['translation_text']
    return translated_text

def translation_pipeline_eng_rus(text):
    # Создание объекта pipeline для перевода текста с английского на русский
    translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
    translated_text = translation_pipeline(text)[0]['translation_text']
    return translated_text

def emotion_pipeline(text, label_mapping):
    # Создание объекта pipeline для классификации эмоций с использованием модели distilbert
    classification_pipeline = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")

    results = classification_pipeline(text, top_k=5)
    formatted_results = {label_mapping.get(result['label'], result['label']): result['score'] for result in results}
    return formatted_results

def summarization_pipeline(text):
    # Создание объекта pipeline для автоматической суммаризации текста
    summarizer = pipeline("summarization", model="slauw87/bart_summarisation")
    summary = summarizer(text, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']