# Otherwise known as a lexical or keyword search.

# %% Download and parse the data
import requests
import pandas as pd

faqs_url = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"
response = requests.get(faqs_url)
faqs_raw = response.json()

print(f"The course is: {faqs_raw[2]['course']}")
print(f"The data from the third question is: \n{faqs_raw[0]['documents'][2]}")

# extract just the text into a list
docs_lst = []
for course in faqs_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        docs_lst.append(doc)
print(f"The data from the third question is: \n{docs_lst[2]}")

# turn the text into a pandas dataframe
docs_df = pd.DataFrame(docs_lst, columns=['course', 'section', 'question',
                                          'text'])
docs_df.rename(columns={'text': 'reply'}, inplace=True)
print(docs_df.tail())

# filter to just a specific course
print(docs_df[docs_df.course == 'mlops-zoomcamp'])

# %% Implement a basic search with a count vectoriser
from sklearn.feature_extraction.text import CountVectorizer

## note that this is a BoW approach where we don't care about word order; only
## the presence/absence of a word
cv = CountVectorizer(stop_words='english')
# vectorise just the replies
cv.fit(docs_df.reply)

# view all the different tokens in the replies
print(cv.get_feature_names_out())
# view the number of distinct tokens
print(cv.get_feature_names_out().shape)

# restrict the tokens to those that appear in at least 5 replies
cv = CountVectorizer(min_df=5, stop_words='english')
cv.fit(docs_df.reply)
print(cv.get_feature_names_out())
# down from ~6500 to ~1300 tokens
print(cv.get_feature_names_out().shape)

# transform the documents into a (sparse) matrix
X_cv = cv.transform(docs_df.reply)
X_cv
print(X_cv.todense())
# convert it into a dataframe
names_cv = cv.get_feature_names_out()
print(pd.DataFrame(X_cv.todense(), columns=names_cv))
mat_df = pd.DataFrame(X_cv.todense(), columns=names_cv).T
# sort by the first document/reply
mat_df.sort_values(by=0, ascending=False, inplace=True)
print(mat_df)

# %% Weighting results by the amount of documents they appear in
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(stop_words='english', min_df=5)
tv.fit(docs_df.reply)
names = tv.get_feature_names_out()

X = tv.transform(docs_df.reply)
# X = tv.fit_transform(docs_df.reply)

docs_df2 = pd.DataFrame(X.toarray(), columns=names).T.round(2)
docs_df2.sort_values(by=0, ascending=False, inplace=True)
print(docs_df2.head())

# %% Test the vectorisers
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

query = "Do I need to know python to sign up for the course?"

# apply vectoriser to query
q_cv = cv.transform([query])
print(q_cv.toarray())

query_dict = dict(zip(names_cv, q_cv.toarray()[0]))
doc_dict = dict(zip(names_cv, X_cv.toarray()[1]))

# compute a summed dot product to identify the similarity of each reply with
# the query (aka costine similarity)
sim_cv = X_cv.dot(q_cv.T).todense()
sim_cv[:10]

# in practice use the cosine_similarity function from scikit-learn
score_cv = cosine_similarity(X_cv, q_cv).flatten()
# sort the similarity scores to give us the document indices in descending
# order
top_idxs_cv = np.argsort(score_cv)[-5:]

for idx in top_idxs_cv:
    print(docs_df.iloc[idx])
    print("\n")

# %% Implement a search across multiple fields
cols = ['section', 'question', 'reply']

matrices = {}
vectorisers = {}

for col in cols:
    v = TfidfVectorizer(stop_words='english', min_df=5)
    X = v.fit_transform(docs_df[col])
    matrices[col] = X
    vectorisers[col] = v

mat_size = len(docs_df)
scores = np.zeros(mat_size)
for col in cols:
    q = vectorisers[col].transform([query])
    X = matrices[col]

    col_score = cosine_similarity(X, q).flatten()
    score = scores + col_score

top_idxs = np.argsort(score)[-5:]
# top_idxs = np.argsort(-score)[:5]
results = docs_df.iloc[top_idxs]

# apply a filter
filters = {'course': 'mlops-zoomcamp'}

for field, value in filters.items():
    mask = (docs_df[field] == value).astype(int)
    masked_score = score * mask

top_idxs = np.argsort(masked_score)[-5:]
results_filtered = docs_df.iloc[top_idxs]

# %% 'Boost' the importance of selected columns
boosts = {'question': 3, 'text': 0.5}
filters = {'course': 'mlops-zoomcamp'}

mat_size = len(docs_df)
scores = np.zeros(mat_size)
for col in cols:
    q = vectorisers[col].transform([query])
    X = matrices[col]

    col_score = cosine_similarity(X, q).flatten()
    boost = boosts.get(col, 1.)
    score = scores + boost * col_score

top_idxs = np.argsort(score)[-5:]
results = docs_df.iloc[top_idxs]

# apply a filter
filters = {'course': 'mlops-zoomcamp'}

for field, value in filters.items():
    mask = (docs_df[field] == value).astype(int)
    masked_score = score * mask

top_idxs = np.argsort(masked_score)[-5:]
results_filtered = docs_df.iloc[top_idxs]


# %% Turn it into a class
class TextSearch:

    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorisers = {}

    def fit(self, records, vectoriser_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            vec = TfidfVectorizer(**vectoriser_params)
            X = vec.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorisers[f] = vec

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorisers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')


# create an instance of the class and list the fields we wish to search over
index = TextSearch(
    text_fields=['section', 'question', 'reply']
)
# apply the vectoriser to the documents
index.fit(docs_df)
# perform the search and return the results in the same format as an
# ElasticSearch i.e. a list of dictionaries
search_results = index.search(
    query="I've just signed up. Is it too late to join the course?",
    n_results=5,
    boost={'question': 3.0},
    filters={'course': 'data-engineering-zoomcamp'}
)

# for a more complete example, see
# https://github.com/alexeygrigorev/minsearch/blob/main/minsearch.py
