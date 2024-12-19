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
    with open('saved_log/log.txt', 'r') as file:
        rows = file.readlines()
        last_row = rows[-1].strip() if rows else None
        if last_row is None:
            print("File is empty.")

        file.close()
    return last_row

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

from transformers import pipeline

def emotions_pipeline(text, label_mapping):
    # Create a pipeline for text classification
    classification_pipeline = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")

    # Perform text classification
    results = classification_pipeline(text, top_k=100)

    # Filter results for specific keys
    filtered_labels = ["joy", "sadness", "anger", "fear", "surprise"]
    filtered_results = {
        label_mapping.get(result['label'], result['label']): result['score']
        for result in results
        if result['label'] in filtered_labels
    }

    # Add percentage symbol to the values
    final_results = {key: f"{round(value * 100, 2)}%" for key, value in filtered_results.items()}

    # Return filtered results with percentages
    return final_results


def emotion_pipeline(text):
    #translated_text = translation_pipeline_rus_eng(result)
    #sentiment_score = sentiment_metrics(translated_text)
    #emotions_scores = emotion_pipeline(translated_text, label_mapping)
    message_content = f'''Сделай небольшой анализ тональности текста и выведи результат. Без цифр или процентов
    Сделай текст в виде списка, делай отступы, не используй ** ** символы
    : {text}.
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

def summarization_pipeline(text, command=None):
    message_content = f'''Представь что ты клиентский мененджер в банке. Сделай краткую суммаризацию следующего текста с учетом этого: {text}.
    Дополнительно {command}
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