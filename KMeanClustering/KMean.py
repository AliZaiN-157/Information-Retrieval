import re
import os
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, adjusted_rand_score

ps = PorterStemmer()
stopwords = []

# Stopwords


def load_stopwords():
    with open('./KMeanClustering/Stopword-List.txt', 'r') as f:
        for line in f:
            # if line is space, skip
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
    for filename in sorted(os.listdir(r'./Resources/ResearchPapers'), key=lambda x: int(x[:-4])):
        with open(r'./Resources/ResearchPapers/' + filename, 'r') as f:
            filename = int(filename[:-4])
            index.append(filename)
            data.append(f.read())
    return data, index


def create_dataframe(data, index):
    df = pd.DataFrame(index=index)
    df['data'] = data
    df['data'] = df['data'].apply(preprocessing)
    df['data'] = df['data'].apply(lambda x: ' '.join(x))
    for i in df.index:
        if i == 1 or i == 2 or i == 3 or i == 7:
            df.at[i, 'gt_label'] = 0  # Explainable Artificial Intelligence
        elif i == 8 or i == 9 or i == 11:
            df.at[i, 'gt_label'] = 1  # Heart Failure
        elif i == 12 or i == 13 or i == 14 or i == 15 or i == 16:
            df.at[i, 'gt_label'] = 2  # "Time Series Forecasting"
        elif i == 17 or i == 18 or i == 21:
            df.at[i, 'gt_label'] = 3  # "Transformer Model"
        elif i == 22 or i == 23 or i == 24 or i == 25 or i == 26:
            df.at[i, 'gt_label'] = 4  # "Feature Selection"
    df['gt_label'] = df['gt_label'].astype(int)
    return df


if os.path.exists("./KMeanClustering/tf_idf.csv"):
    df = pd.read_csv("./KMeanClustering/tf_idf.csv")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['data'])
    X = X.toarray()
else:
    data, index = load_data()
    df = create_dataframe(data, index)
    df.to_csv("./KMeanClustering/tf_idf.csv", index=False)


def kmean(X, n_clusters, df):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                    max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    df['cluster'] = labels
    return kmeans, df


def pca_transform(X, df):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    df['pca3'] = X_pca[:, 2]
    return df

# Evaluation


def purity_score(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def evaluation(true_labels, cluster_labels):
    true_labels = df['gt_label']
    cluster_labels = df['cluster']
    purity = purity_score(true_labels, cluster_labels)
    silhouette = silhouette_score(X, cluster_labels)
    random_index = adjusted_rand_score(true_labels, cluster_labels)
    return purity, silhouette, random_index


def elbow_method(X, true_labels):
    wcss = []
    sil = []
    rand = []
    for i in range(2, 10):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric='euclidean'))
        rand.append(adjusted_rand_score(true_labels, labels))

    return wcss, sil, rand


def set_cluster(df=df, n_clusters=5):
    kmeans, df = kmean(X, n_clusters, df)
    true_labels = df['gt_label']
    cluster_labels = df['cluster']
    df = pca_transform(X, df)
    purity, silhouette, RI = evaluation(true_labels, cluster_labels)
    wcss, sil, rand = elbow_method(X, true_labels)
    return df, kmeans, wcss, sil, rand, purity, silhouette, RI
