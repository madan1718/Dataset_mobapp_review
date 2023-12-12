# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:36:51 2023

@author: Madanjit
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load and preprocess the training data (CSV File 1)
train_df = pd.read_csv('training.csv')
X_train = train_df['Review Text']
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['Category'])
print(y_train)
tfidf_vectorizer = TfidfVectorizer(max_features=100000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Load and preprocess the testing data (CSV File 2)
test_df = pd.read_csv('test.csv')
X_test = test_df['Review Text']

# Check for and handle missing values in testing data
X_test = X_test.fillna('')

# Ensure data type compatibility with TfidfVectorizer
X_test = X_test.astype(str)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_tfidf)
print(y_pred)
# Ensure that labels match the encoded labels from training data
test_labels = label_encoder.transform(test_df['Category'])

# Evaluate the model's performance on the testing data
accuracy = accuracy_score(test_labels, y_pred)

# Specify the correct class labels in target_names based on your dataset
target_names = ['usability', 'others','security and privacy'] 
print(target_names) # Adjust these labels to match your dataset
print(test_labels)

classification_rep = classification_report(test_labels, y_pred, target_names=target_names)

# Print the classification report and accuracy
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Save the results to a new CSV file
test_df['Predicted_Category'] = label_encoder.inverse_transform(y_pred)
test_df.to_csv('results.csv', index=False)
