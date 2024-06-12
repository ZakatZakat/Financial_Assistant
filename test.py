from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

with open('saved_log/log.txt', 'r') as file:
    rows = file.readlines()
    last_row = rows[-1].strip() if rows else None
    if last_row is None:
        print("File is empty.")

    file.close()

input_ids = tokenizer.encode(last_row, return_tensors="pt")
translation = model.generate(input_ids)
translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

# Load the pre-trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('joeddav/distilbert-base-uncased-go-emotions-student')
model = DistilBertForSequenceClassification.from_pretrained('joeddav/distilbert-base-uncased-go-emotions-student')

# Encode a text input
input_text = translated_text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate an output
# Generate an output
with torch.no_grad():
    output = model(input_ids)
    logits = output.logits
    top_k_values, top_k_indices = torch.topk(logits, k=5, dim=-1)
    
    emotions = ['admiration', 'amusement', 'anger', 'bother', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disapproval', 
                'disappointment', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 
                'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    
    predicted_emotions_scores = []
    for i in range(top_k_indices.size(0)):
        predicted_emotions = [emotions[idx] for idx in top_k_indices[i]]
        predicted_scores = top_k_values[i].tolist()
        formatted_emotions_scores = {emotion: score for emotion, score in zip(predicted_emotions, predicted_scores)}
        predicted_emotions_scores.append(formatted_emotions_scores)
    
    for i, emotions_scores in enumerate(predicted_emotions_scores):
        print(f"Top 5 Predicted Emotions:")
        for emotion, score in emotions_scores.items():
            print(f"{emotion}: {score:.4f}")


