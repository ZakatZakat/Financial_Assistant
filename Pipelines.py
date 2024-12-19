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
    return last_row#, label_mapping

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

def emotion_pipeline(text):
    #translated_text = translation_pipeline_rus_eng(result)
    #sentiment_score = sentiment_metrics(translated_text)
    #emotions_scores = emotion_pipeline(translated_text, label_mapping)
    message_content = f'''Определи тональность данного текста и распредели его согласно этим эмоциональным коэффициентам
    1.	Радость (joy)
	2.	Грусть (sadness)
	3.	Гнев (anger)
	4.	Страх (fear)
	5.	Удивление (surprise)
    
    Выведи и подбери коэффициента сам, чтобы сумма их была 100%, а они распределялись на основе твоего анализа
    Пример: 
    
    Данный текст в целом имеет позитивный настрой, хотя присутствуют и упоминания о проблемах. Проанализировав текст, можно выделить следующие эмоциональные коэффициенты:
	1.	Радость (joy) – 50%
	2.	Грусть (sadness) – 10%
	3.	Гнев (anger) – 0%
	4.	Страх (fear) – 5%
	5.	Удивление (surprise) – 5%

    Остальные 30% остаются неопределёнными, так как в тексте нет явных признаков других эмоций. Таким образом, итоговое распределение будет следующим:
	•	Радость: 50%
	•	Грусть: 10%
	•	Гнев: 0%
	•	Страх: 5%
	•	Удивление: 5%
    Сделай текст читабельным, делай отступы, не используй ** ** символы
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

def summarization_pipeline(text):
    message_content = f'''Представь что ты клиентский мененджер в банке. Сделай подробную суммаризацию следующего текста с учетом этого: {text}.
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