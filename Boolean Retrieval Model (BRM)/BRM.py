import os
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

nltk.download('punkt')

dictionary = {}
stopwords = []
ps = PorterStemmer()

# loading the stop words


def load_stopwords():
    with open('Stopword-List.txt', 'r') as f:
        for line in f:
            # if line is space, skip
            if not line.strip():
                continue
            stopwords.append(line.strip())


load_stopwords()

# Preprocessing


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

# loading the corpuses


def load_corpuses():
    lines_with_doc_id = {}
    corpus = {}
    for filename in os.listdir('ResearchPapers'):
        with open('ResearchPapers/' + filename, 'r') as f:
            lines = f.read().replace("\n", " ")
            # remove .txt from filename
            filename = filename[:-4]
            lines_with_doc_id[int(filename)] = lines
            # lines.append(f.read())
            tokens = preprocessing(lines)
            term_frequency = Counter(tokens)
            for term, frequency in term_frequency.items():
                if term not in corpus:
                    corpus[term] = [frequency, {int(filename): []}]
                else:
                    corpus[term][0] += frequency
                    corpus[term][1][int(filename)] = []
            for i, token in enumerate(tokens):
                corpus[token][1][int(filename)].append(i)
            # finally, sort the dictionary by key
            corpus = dict(sorted(corpus.items()))
    return corpus, lines_with_doc_id

# Function to find postings of a term


def find_postings(term):
    if term in corpus:
        return set(corpus[term][1].keys())
    return []

# write dictionary and doc ids to file


def write_to_file():
    corpus, doc_ids = load_corpuses()
    with open('Final_Dictionary.txt', 'w') as f:
        f.write(json.dumps(corpus))
    with open('DocID.txt', 'w') as f:
        f.write(json.dumps(doc_ids))
    print("Dictionary saved to file")

# load dictionary from file


def load_dictionary():
    with open('Final_Dictionary.txt', 'r') as f:
        return json.loads(f.read())

# load doc ids from file


def load_doc_ids():
    with open('DocID.txt', 'r') as f:
        return json.loads(f.read())


# if Final_Dictionary.txt exists, load it
if os.path.exists('Final_Dictionary.txt') and os.path.exists('DocID.txt'):
    print("Loading dictionary from file")
    corpus = load_dictionary()
    doc_ids = load_doc_ids()
else:
    corpus, doc_ids = load_corpuses()
    write_to_file()

# Function to solve positional query


def positional_query(term1, term2, k):
    postings1 = find_postings(term1)
    postings2 = find_postings(term2)
    result = set()
    for doc_id in postings1:
        if doc_id in postings2:
            positions1 = corpus[term1][1][doc_id]
            positions2 = corpus[term2][1][doc_id]
            for pos1 in positions1:
                for pos2 in positions2:
                    if abs(pos1 - pos2) <= k:
                        result.add(doc_id)
                        break
    return result

# Function to solve boolean operators


def solve_boolean_operators(query):
    result = set()
    operator = None
    for token in query:
        if token == 'and':
            operator = 'and'
        elif token == 'or':
            operator = 'or'
        elif token == 'not':
            operator = 'not'
        else:
            if operator == 'and':
                result = result.intersection(find_postings(token))
            elif operator == 'or':
                result = result.union(find_postings(token))
            elif operator == 'not':
                result = set(doc_ids.keys()) - find_postings(token)
            else:
                result = find_postings(token)
            operator = None

    return result

# Function to fetch the query from the user and return the result


def queryFetcher(user_input):
    # user_input = input("Enter your query: ")
    if '/' in user_input:
        query = user_input.replace("/", "")
        term1, term2, k = query.split()
        term1 = ps.stem(term1)
        term2 = ps.stem(term2)
        result = positional_query(term1, term2, int(k))
    else:
        query = user_input.lower()
        query = query.split()
        query = [ps.stem(token) for token in query]
        result = solve_boolean_operators(query)
    result = [int(i) for i in result]
    result.sort()
    return result
