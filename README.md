# Movie Review Sentiment Classifier

Simple ML project that classifies a movie review as positive or negative.
Trained on the IMDB 50k reviews dataset using TF-IDF + Logistic Regression, with a small Streamlit app on top to test it interactively.

## How it works

1. `train_model.py` — loads the dataset, cleans the text (strips HTML tags and punctuation), converts it to TF-IDF vectors, trains a Logistic Regression model, and saves both the model and the vectorizer as pickle files.
2. `app.py` — loads those saved files and gives you a text box in the browser to type a review and get a prediction with a confidence score.

## Results

On the held-out test set:
- Accuracy: ~89%
- Precision: ~89%
- Recall: ~89%

(numbers will vary slightly depending on train/test split)

## Running it yourself

1. Download the dataset from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), place `IMDB Dataset.csv` in this folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train the model:
   ```
   python train_model.py
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## Why TF-IDF + Logistic Regression instead of something fancier

Wanted something that's fast to train, easy to explain, and interpretable — TF-IDF + Logistic Regression is a solid baseline for text classification before reaching for embeddings or deep learning. Main limitation: it doesn't understand context or sarcasm, since it's purely frequency-based.

## Possible improvements

- Try Naive Bayes and compare against Logistic Regression
- Use n-grams (bigrams/trigrams) instead of single words
- Swap TF-IDF for sentence embeddings for better context understanding
