from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.datasets import imdb
import numpy as np
from preprocess_text import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

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
X_train_full = vectorizer.fit_transform(train_reviews)
X_test = vectorizer.transform(test_reviews)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, train_labels, test_size=0.2, random_state=42)

# Create and train the logistic regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = logistic_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Save the model and vectorizer
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
