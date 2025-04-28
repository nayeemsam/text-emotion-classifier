
Text Emotion Detector

This project fine-tunes a pre-trained j-hartmann/emotion-english-distilroberta-base model using the GoEmotions dataset by Google to better detect emotions from text inputs.

The goal is to build a text emotion detection system that can help in areas like mental health monitoring, user sentiment analysis, and stress detection.

Model Details

Base Model: j-hartmann/emotion-english-distilroberta-base

Dataset Used: GoEmotions Dataset

Fine-Tuned for: 28 emotion classes (including anger, joy, sadness, fear, etc.)


How to Use

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("nayeemsam/my-goemotions-finetuned")
tokenizer = AutoTokenizer.from_pretrained("nayeemsam/my-goemotions-finetuned")

# Sample input
text = "I am feeling so frustrated and angry!"

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

# Predict
outputs = model(**inputs)
logits = outputs.logits
predicted_class_id = logits.argmax(dim=-1).item()

# Class Labels
id2label = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'caring',
    6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment', 10: 'disapproval',
    11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude',
    16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 20: 'optimism', 21: 'pride',
    22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
}

# Output prediction
print(f"Predicted emotion: {id2label[predicted_class_id]}")

Project Stage

This project is in its initial phase. Future goals include:

Improving model accuracy

Building a real-time web or mobile application

Integrating into mental health support systems


Model Link

You can find and use the model on Hugging Face:
Hugging Face Model: nayeemsam/my-goemotions-finetuned

License

Apache-2.0 License


---
