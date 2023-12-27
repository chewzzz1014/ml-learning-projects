import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    train_data = pd.read_csv('../datasets/train.csv')
    test_data = pd.read_csv('../datasets/test.csv')

    X_train = train_data['text']
    y_train = train_data['labels']
    X_test = test_data['text']
    y_test = test_data['labels']

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train_vectorized, y_train)

    # performance
    y_pred = classifier.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

    return classifier

