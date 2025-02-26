import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    data = pd.read_csv('data/Hotel_Reviews.csv')

    #reviews = data["Positive_Review"]
    reviews = data["Negative_Review"]

    tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    X = tfidf_vectorizer.fit_transform(reviews)
    y = data["Reviewer_Score"].map(lambda x: 0 if x < 5.0 else 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

