from transformers import pipeline
import os
from openai import OpenAI
import pprint
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)


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

def translation_pipeline_rus_eng(text):
    translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")
    translated_text = translation_pipeline(text)[0]['translation_text']
    return translated_text

def translation_pipeline_eng_rus(text):
    translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
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

def summarization_pipeline(text):
    message_content = f'''Для полученного текста {text} сформулируй отчет и распиши, о чем был данный переговор менеджера и клиента
    '''
        
        # Отправка запроса к API
    chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            model="gpt-4o-mini",
        )
        
        # Вывод и сохранение результата
    result_content = chat_completion.choices[0].message.content
    
    return result_content