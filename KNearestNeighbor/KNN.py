import re
import os
import pandas as pd
import joblib


from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import warnings
warnings.filterwarnings("ignore")


ps = PorterStemmer()
stopwords = []

# Stopwords


def load_stopwords():
    with open('./KNearestNeighbor/Stopword-List.txt', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            stopwords.append(line.strip())


load_stopwords()

# Preprocess


def preprocessing(corpus):
    # lowercase
    corpus = corpus.lower()
    # remove punctuation
    corpus = re.sub(r'[^\w\s]', '', corpus)
    # remove numbers
    corpus = re.sub(r'\d+', '', corpus)
    # replace multiple spaces with single space
    corpus = re.sub(r'\s+', ' ', corpus)
    # remove leading and trailing spaces
    corpus = corpus.strip()
    # remove irrelevant characters
    corpus = re.sub(r'[^\x00-\x7F]+', '', corpus)
    tokens = word_tokenize(corpus)
    # remove stopwords and stem
    tokens = [ps.stem(token) for token in tokens if token not in stopwords]
    # remove single character tokens
    tokens = [token for token in tokens if len(token) > 1]
    # remove large tokens
    tokens = [token for token in tokens if len(token) < 20]
    # remove tokens with consecutive characters
    tokens = [token for token in tokens if not re.match(
        r".*(.)\1{2,}.*", token)]
    # remove urls with http or https using startswith
    tokens = [token for token in tokens if not token.startswith(
        'http') and not token.startswith('https')]
    # http or https in the middle of the url
    tokens = [token for token in tokens if not re.match(
        r"[a-zA-Z0-9\./]+http[a-zA-Z0-9\./]+", token)]
    # remove url with github
    tokens = [token for token in tokens if not re.match(
        r"github/[a-zA-Z0-9\./]+", token)]
    # remove email addresses using regex
    tokens = [token for token in tokens if not re.match(
        r"[^@]+@[^@]+\.[^@]+", token)]
    return tokens


def load_data():
    data = []
    index = []
    for filename in sorted(os.listdir(r'./Resources//ResearchPapers'), key=lambda x: int(x[:-4])):
        with open(r'./Resources/ResearchPapers/' + filename, 'r') as f:
            filename = int(filename[:-4])
            index.append(filename)
            data.append(f.read())

    df = pd.DataFrame(index=index)
    df['data'] = data
    df['data'] = df['data'].apply(preprocessing)
    df['data'] = df['data'].apply(lambda x: ' '.join(x))

    for i in df.index:
        if i == 1 or i == 2 or i == 3 or i == 7:
            df.at[i, 'class'] = "Explainable Artificial Intelligence"
        elif i == 8 or i == 9 or i == 11:
            df.at[i, 'class'] = "Heart Failure"
        elif i == 12 or i == 13 or i == 14 or i == 15 or i == 16:
            df.at[i, 'class'] = "Time Series Forecasting"
        elif i == 17 or i == 18 or i == 21:
            df.at[i, 'class'] = "Transformer Model"
        elif i == 22 or i == 23 or i == 24 or i == 25 or i == 26:
            df.at[i, 'class'] = "Feature Selection"

    return df


def KNN(df):
    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(df['data'])
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=32)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    score = cross_val_score(knn, X, y, cv=5)
    print('Cross Validation Score:', score.mean())

    # predict
    y_pred = knn.predict(X_test)

    # accuracy
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1:', f1)

    return knn, tfidf


if os.path.exists('./KNearestNeighbor/KNN_model.pkl') and os.path.exists('./KNearestNeighbor/KNN_tfidf.pkl'):
    print('Loading model ...')
    knn = joblib.load('./KNearestNeighbor/KNN_model.pkl')
    tfidf = joblib.load('./KNearestNeighbor/KNN_tfidf.pkl')
else:
    print('Training model ...')
    df = load_data()
    knn, tfidf = KNN(df)
    joblib.dump(knn, './KNearestNeighbor/KNN_model.pkl')
    joblib.dump(tfidf, './KNearestNeighbor/KNN_tfidf.pkl')


def predict(text):
    text = preprocessing(text)
    text = ' '.join(text)
    text = tfidf.transform([text])
    return knn.predict(text)[0]
