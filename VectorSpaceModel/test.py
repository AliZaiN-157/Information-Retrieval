import os
import re
import math
import pandas as pd
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
stopwords = []
N = 20


def load_stopwords():
    with open('Stopword-List.txt', 'r') as f:
        for line in f:
            # if line is space, skip
            if not line.strip():
                continue
            stopwords.append(line.strip())


load_stopwords()


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


doc_ids = []


def load_data():
    data = []
    for filename in sorted(os.listdir(r'../ResearchPapers'), key=lambda x: int(x[:-4])):
        with open(r'../ResearchPapers/' + filename, 'r') as f:
            doc_ids.append(int(filename[:-4]))
            data.append(f.read())
    return data


data = load_data()

index = ["terms", *doc_ids, 'df', 'idf']
df = pd.DataFrame(columns=index)
df.head()


def compute_tf_idf(df, data):
    for i, doc in enumerate(data):
        tokens = preprocessing(doc)
        token_count = Counter(tokens)
        for token, count in token_count.items():
            if token not in df['terms'].values:
                df = df._append({'terms': token}, ignore_index=True)
            df.loc[df['terms'] == token, doc_ids[i]] = count
    df = df.fillna(0)
    df['df'] = df[doc_ids].apply(lambda x: sum(x > 0), axis=1)
    df['idf'] = df['df'].apply(lambda x: math.log10(N/x))
    for doc_id in doc_ids:
        # df[doc_id] = df[doc_id].apply(lambda x: 1 + math.log10(x) if x > 0 else 0)
        df[doc_id] = df[doc_id] * df['idf']
    return df


if os.path.exists('tf_idf.csv'):
    print('Loading index from file')
    new_df = pd.read_csv('tf_idf.csv')
else:
    print('Computing index')
    new_df = compute_tf_idf(df, data)
    new_df.to_csv('tf_idf.csv', index=False)
    print('Index saved to file')


def add_query_tf_idf(query):
    global new_df
    tokens = preprocessing(query)
    token_count = Counter(tokens)
    for token, count in token_count.items():
        if token not in new_df['terms'].values:
            new_df = new_df._append({'terms': token}, ignore_index=True)
        new_df.loc[new_df['terms'] == token, 'query'] = count
    new_df = new_df.fillna(0)
    # new_df['query'] = new_df['query'].apply(
    #     lambda x: 1 + math.log(x, 10) if x > 0 else 0)
    new_df['query'] = new_df['query'] * new_df['idf']
    return new_df


# def create_vector():
#     global new_df
#     float_cols = new_df.select_dtypes('float64').columns
#     vector = {}
#     for col in float_cols:
#         vector[col] = new_df[col].values
#     vector.pop('idf')
#     query_vector = vector.pop('query')
#     new_df = new_df.drop('query', axis=1)
#     # print(new_df.head(10))
#     return vector, query_vector

def create_vector():
    global new_df
    vec = {}
    for id in doc_ids:
        vec[id] = new_df[str(id)].values
    # make the query column fill with 0
    query_vector = new_df["query"].values
    new_df['query'] = new_df['query'].fillna(0)
    return vec, query_vector


def cosine_similarity(query_vector, doc_vector):
    dot_product = sum(query_vector * doc_vector)
    query_norm = math.sqrt(sum(query_vector ** 2))
    doc_norm = math.sqrt(sum(doc_vector ** 2))
    return dot_product / (query_norm * doc_norm)


def queryFetcher(query):
    print('Processing query:', query)
    # query = "information retrieval"
    new_df = add_query_tf_idf(query)
    vector, query_vector = create_vector()
    # calculate cosine similarity
    cosine_sim = {}
    for doc_id, doc_vector in vector.items():
        cosine_sim[doc_id] = cosine_similarity(query_vector, doc_vector)
    # sort the dictionary by values by a threshold of 0.03
    cosine_sim = {k: v for k, v in sorted(
        cosine_sim.items(), key=lambda item: item[1], reverse=True) if v > 0.03}
    return list(cosine_sim.keys())


print(queryFetcher("machine learning"))
print(queryFetcher("intelligent search"))
print(queryFetcher("cancer"))
print(queryFetcher("deep convolutional network"))
print(queryFetcher("artificial intelligence"))


# machine learning ['24', '7', '16', '2', '1']
# intelligent search ['7', '3', '1', '2']
# cancer NIL
# deep convolutional network ['16', '3', '2', '7']
# artificial intelligence ['1', '8']
# transformer ['21', '18']
# local global feature ['22', '23', '24', '25', '26', '7']
# feature selection machine learning ['22', '24', '23', '25', '26', '7', '1']
# information retrieval ['1']
# natural intelligence ['7', '2', '3', '1']
