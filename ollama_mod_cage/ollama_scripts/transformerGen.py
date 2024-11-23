""" transformerGen.py

    This is a small python transformer script generated with qwen2.5-coder in the oarc webui.
    @Leo_Borcherding 11/23/2024
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline

# Load the tokenizer and model
model_name = "gpt2"  # You can choose other models like "gpt3", "bert-large-cased", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load a sentiment analysis pipeline
sentiment_analysis_pipeline = pipeline("sentiment-analysis")

# Function to generate a response from the chatbot
def get_response(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to analyze sentiment of the input text
def get_sentiment(text):
    result = sentiment_analysis_pipeline(text)
    return result[0]['label']

# Function to personalize greetings based on the sentiment
def personalize_greeting(sentiment):
    if sentiment == "POSITIVE":
        return "Hello! How can I assist you today?"
    elif sentiment == "NEGATIVE":
        return "I'm sorry to hear that. Let's try to make things better!"
    else:
        return "Hi there! Feel free to ask me anything."

# Example conversation
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Get sentiment of the user input
    user_sentiment = get_sentiment(user_input)
    
    # Generate personalized greeting based on sentiment
    greeting = personalize_greeting(user_sentiment)
    print(greeting)
    
    bot_response = get_response(user_input)
    print(f"Bot: {bot_response}")
