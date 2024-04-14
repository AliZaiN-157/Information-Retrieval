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
    with open('./VectorSpaceModel/Stopword-List.txt', 'r') as f:
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
            filename = int(filename[:-4])
            doc_ids.append(filename)
            data.append(f.read())
    return data


def compute_tf_idf(data):
    index = ["terms", *doc_ids, 'df', 'idf']
    df = pd.DataFrame(columns=index)
    for i, doc in enumerate(data):
        tokens = preprocessing(doc)
        token_count = Counter(tokens)
        # add tokens to the dataframe
        for token, count in token_count.items():
            # if token not in the dataframe, add it
            if token not in df['terms'].values:
                df = df._append({'terms': token}, ignore_index=True)
            # add count to the corresponding doc_id
            df.loc[df['terms'] == token, doc_ids[i]] = count
    df = df.fillna(0)
    # calculating df
    df['df'] = df[doc_ids].apply(lambda x: sum(x > 0), axis=1)
    # calculating idf
    df['idf'] = df['df'].apply(lambda x: math.log(N/x))
    # calculating tf-idf
    for doc_id in doc_ids:
        # df[doc_id] = df[doc_id].apply(
        #     lambda x: 1 + math.log(x, 10) if x > 0 else 0)
        df[doc_id] = df[doc_id] * df['idf']
    return df


if os.path.exists('./VectorSpaceModel/tf_idf.csv'):
    print('Loading index from file')
    new_df = pd.read_csv('./VectorSpaceModel/tf_idf.csv')
else:
    print('Loading data')
    data = load_data()
    print('Computing index')
    new_df = compute_tf_idf(data)
    new_df.to_csv('tf_idf.csv')
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


def create_vector():
    global new_df
    float_cols = new_df.select_dtypes('float64').columns
    vector = {}
    for col in float_cols:
        vector[col] = new_df[col].values
    # if df in vector, remove it
    if 'df' in vector:
        vector.pop('df')
    vector.pop('idf')
    query_vector = vector.pop('query')
    new_df = new_df.drop('query', axis=1)
    return vector, query_vector


def cosine_similarity(query_vector, doc_vector):
    dot_product = sum(query_vector * doc_vector)
    query_norm = math.sqrt(sum(query_vector ** 2))
    doc_norm = math.sqrt(sum(doc_vector ** 2))
    return dot_product / (query_norm * doc_norm)


def queryFetcher(query):
    print('Processing query:', query)
    new_df = add_query_tf_idf(query)
    vector, query_vector = create_vector()
    cosine_sim = {}
    for doc_id, doc_vector in vector.items():
        cosine_sim[doc_id] = cosine_similarity(query_vector, doc_vector)
    cosine_sim = {k: v for k, v in sorted(
        cosine_sim.items(), key=lambda item: item[1], reverse=True) if v > 0.03}
    return list(cosine_sim.keys())
