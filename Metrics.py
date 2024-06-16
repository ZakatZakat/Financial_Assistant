from Pipelines import preprocess, sentiment_tempreture

def calculate_normalized_metric(sentiment_scores):
    # Определение весов для каждого настроения
    weights = {
        'negative': -1,
        'neutral': 0,
        'positive': 1
    }
    
    # Вычисление взвешенной суммы сентимента
    weighted_sum_scores = sum(weights[sentiment['label']] * sentiment['score'] for sentiment in sentiment_scores)
    
    # Определение минимального и максимального значений сентимента
    min_sentiment = -1
    max_sentiment = 1
    
    # Нормализация сентимента к масштабу [0, 1]
    normalized_metric = (weighted_sum_scores - min_sentiment) / (max_sentiment - min_sentiment)
    return normalized_metric

def sentiment_metrics(result):
    sentiment_scores = sentiment_tempreture(result)

    normalized_metric = calculate_normalized_metric(sentiment_scores)
    return normalized_metric

