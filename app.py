from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import SentimentRecommenderModel

# import os
# Get and print the current working directory
# print("Current Working Directory:", os.getcwd())


app = Flask(__name__)

# Load model
try:
    sentiment_model = SentimentRecommenderModel()
except Exception as e:
    print(f"Error initializing sentiment model: {e}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    try:
        user = request.form['userName'].lower()
        items = sentiment_model.getSentimentRecommendations(user)

        if items is not None:
            return render_template(
                "index.html", 
                column_names=items.columns.values, 
                row_data=list(items.values.tolist()), 
                zip=zip)
        else:
            return render_template("index.html", message="User not found!")
    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template("index.html", message="An error occurred during prediction.")


@app.route('/predictSentiment', methods=['POST'])
def predict_sentiment():
    try:
        review_text = request.form["reviewText"]
        pred_sentiment = sentiment_model.classify_sentiment(review_text)
        return render_template("index.html", sentiment=pred_sentiment)
    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        return render_template("index.html", message="An error occurred during sentiment prediction.")


if __name__ == '__main__':
    app.run()