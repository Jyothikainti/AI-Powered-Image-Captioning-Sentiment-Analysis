from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ✅ Custom Sentiment Analysis using VADER
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()

    # Custom word weighting for aggression
    aggression_words = ["fight", "battle", "attack", "claws", "ferocity", "roar", "dominance", "aggressive", "wild", "brutal"]
    for word in aggression_words:
        if word in text.lower():
            return "NEGATIVE"  # Force negative sentiment for aggressive descriptions

    sentiment_score = analyzer.polarity_scores(text)['compound']
    
    if sentiment_score >= 0.05:
        return "POSITIVE"
    elif sentiment_score <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"
