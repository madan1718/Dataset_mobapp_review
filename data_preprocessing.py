# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:30:23 2023

@author: Madanjit
"""

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from googletrans import Translator

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file
input_csv_file = "C:/Users/Madanjit/Reviews/beatswiper.csv"
output_csv_file = "beatswiper_pre_data.csv"

data = pd.read_csv(input_csv_file)

# Define a function for text translation to English
def translate_to_english(text):
    translator = Translator()
    translated_text = translator.translate(text, src='auto', dest='en').text
    return translated_text

# Define functions for text preprocessing
def preprocess_text(text):
    # Remove special characters, numbers, and extra whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lowercase the text
    text = text.lower()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

def perform_stemming(text):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    return " ".join(stemmed_words)

def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(lemmatized_words)

# Apply translation to English to the 'Text' column
data['Review Text'] = data['Review Text'].apply(translate_to_english)

# Apply preprocessing to the translated text
data['Review Text'] = data['Review Text'].apply(preprocess_text)

# Uncomment one of the following lines to choose stemming or lemmatization
# data['Text'] = data['Text'].apply(perform_stemming)
# data['Text'] = data['Text'].apply(perform_lemmatization)

# Save the preprocessed data to a new CSV file
data.to_csv(output_csv_file, index=False)
