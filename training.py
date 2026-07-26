import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import re
import pickle

# dataset: IMDB Dataset.csv from Kaggle (50k reviews, columns are "review" and "sentiment")
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
df = pd.read_csv("IMDB Dataset.csv")


def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)  # dataset has a lot of <br /> tags in it
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # drop numbers/punctuation
    text = text.lower().strip()
    return text


df["clean_review"] = df["review"].apply(clean_text)

X = df["clean_review"]
y = df["sentiment"].map({"positive": 1, "negative": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print("accuracy :", accuracy_score(y_test, preds))
print("precision:", precision_score(y_test, preds))
print("recall   :", recall_score(y_test, preds))
print(confusion_matrix(y_test, preds))

# save both, need the vectorizer at inference time too, not just the model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("saved model + vectorizer")
