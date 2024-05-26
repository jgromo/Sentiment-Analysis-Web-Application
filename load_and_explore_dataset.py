from keras.datasets import imdb
import numpy as np
from preprocess_text import preprocess_text  # Import the preprocessing function

# Load the dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Get the word index from the dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Preprocess and decode a sample review
sample_review = decode_review(train_data[0])
print("Original review:", sample_review)
preprocessed_review = preprocess_text(sample_review)
print("Preprocessed review:", preprocessed_review)

# Explore the dataset
print("Number of training samples:", len(train_data))
print("Number of test samples:", len(test_data))
print("First training label:", train_labels[0])

# Check the length of the reviews
review_lengths = [len(review) for review in train_data]
average_length = np.mean(review_lengths)
max_length = np.max(review_lengths)
min_length = np.min(review_lengths)

print("Average review length:", average_length)
print("Maximum review length:", max_length)
print("Minimum review length:", min_length)
