from transformers import pipeline

# Create a pipeline for sequence-to-sequence translation
translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

# Create a pipeline for text classification
classification_pipeline = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")

# Define a dictionary to map English labels to Russian labels
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

# Perform sequence-to-sequence translation
translated_text = translation_pipeline(last_row)[0]['translation_text']

# Perform text classification
results = classification_pipeline(translated_text, top_k=5)

# Print the top 5 classification results with Russian labels
print("Top 5 Classification Results:")
for result in results:
    label = label_mapping.get(result['label'], result['label'])
    print(f"Label: {label}, Score: {result['score']}")