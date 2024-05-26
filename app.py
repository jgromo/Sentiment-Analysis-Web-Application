from flask import Flask, request, render_template
import joblib
from preprocess_text import preprocess_text

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        preprocessed_review = preprocess_text(review)
        vectorized_review = vectorizer.transform([preprocessed_review])
        prediction = model.predict(vectorized_review)
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        return render_template('result.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
