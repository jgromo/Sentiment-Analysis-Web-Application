from keras.datasets import imdb
import numpy as np
from preprocess_text import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Get the word index from the dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Preprocess and decode all reviews
train_reviews = [' '.join([reverse_word_index.get(i, '?') for i in review]) for review in train_data]
test_reviews = [' '.join([reverse_word_index.get(i, '?') for i in review]) for review in test_data]

train_reviews = [preprocess_text(review) for review in train_reviews]
test_reviews = [preprocess_text(review) for review in test_reviews]

# Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_reviews)
X_test = vectorizer.transform(test_reviews)

# Print some information about the transformed data
print("Shape of training data:", X_train.shape)
print("Shape of test data:", X_test.shape)
