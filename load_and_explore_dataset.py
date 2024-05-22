from keras.datasets import imdb
import numpy as np

# Load the dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Get the word index from the dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Decode and print a sample review
print("Decoded review:", decode_review(train_data[0]))

# Explore the dataset
print("Number of training samples:", len(train_data))
print("Number of test samples:", len(test_data))
print("First training label:", train_labels[0])

# Check the length of the reviews
review_lengths = [len(review) for review in train_data]
print("Average review length:", np.mean(review_lengths))
print("Maximum review length:", np.max(review_lengths))
print("Minimum review length:", np.min(review_lengths))
