# Otherwise known as semantic search. Useful when the words don't match
# exactly

# %% Download and parse the data
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

faqs_url = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"
response = requests.get(faqs_url)
faqs_raw = response.json()

docs_lst = []
for course in faqs_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        docs_lst.append(doc)

# turn the text into a pandas dataframe
docs_df = pd.DataFrame(docs_lst, columns=['course', 'section', 'question',
                                          'text'])
docs_df.rename(columns={'text': 'reply'}, inplace=True)

tv = TfidfVectorizer(stop_words='english', min_df=5)
X = tv.fit_transform(docs_df.reply)

cols = ['section', 'question', 'reply']
matrices = {}
vectorisers = {}

for col in cols:
    v = TfidfVectorizer(stop_words='english', min_df=5)
    X = v.fit_transform(docs_df[col])
    matrices[col] = X
    vectorisers[col] = v

# %% Create BoW vector embeddings
# embeddings are a dense representation of a document
from sklearn.decomposition import TruncatedSVD, NMF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. SVD is the simplest approach
# tries to group together words with the same meaning into the same dimension
X = matrices['reply']
vec = vectorisers['reply']

svd = TruncatedSVD(n_components=16)
X_emb = svd.fit_transform(X)
# view the shape of the output array - note the same number of documents but
# reduced number of columns
print(X_emb.shape)
# view the first document
print(X_emb[0])

# apply the same approach to the query
query = "I've just signed up. Is it too late to join the course?"

q = vec.transform([query])
q_emb = svd.transform(q)
print(q_emb[0])

# compute the similarity between the query and the first document
print(np.dot(X_emb[0], q_emb[0]))
# compute the similarity between the query and all documents
score = cosine_similarity(X_emb, q_emb).flatten()
idx = np.argsort(-score)[:10]
print(list(docs_df.loc[idx].reply))

results = docs_df.loc[idx]
print(results)

# 2. NMF is the same idea of SVD, but only produces positive values
nmf = NMF(n_components=16)
X_emb = nmf.fit_transform(X)
print(X_emb[0])

q = vec.transform([query])
q_emb = nmf.transform(q)
print(q_emb[0])

score = cosine_similarity(X_emb, q_emb).flatten()
idx = np.argsort(-score)[:10]
results = docs_df.loc[idx]
print(results)

# %% Create vector embeddings with BERT
import torch
from transformers import BertModel, BertTokenizer

# download the model and the associated tokeniser
tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval() # sets the model to evaluation mode if not training

# tokenise the input text
texts = docs_df.loc[docs_df.course == 'mlops-zoomcamp'].reply.tolist()
encoded_input = tokeniser(texts, padding=True, truncation=True,
                          return_tensors='pt')
print(encoded_input)

# compute the embeddings (feeding it into the neural network)
with torch.no_grad(): # disable gradient calculation for inference (supposedly
                      # makes it faster)
    outputs = model(**encoded_input)
    hidden_states = outputs.last_hidden_state

# and view them
print(hidden_states.shape)
print(hidden_states[0])

# compress the embeddings by calculating the sum/average of the rows for each
# input
sentence_embeddings = hidden_states.mean(dim=1)
print(sentence_embeddings.shape)
print(sentence_embeddings)

# for this example, BERT is overkill as the dimensionality is reasonably high
# BUT is better/more powerful for more complex applications

# convert it into a numpy array
X_emb = sentence_embeddings.numpy()

# move the tensors to the CPU
sentence_embeddings_cpu = sentence_embeddings.cpu()


# put it all into a function
# process the text in batches
def make_batches(seq, n):
    result = []
    for i in range(0, len(seq), n):
        batch = seq[i:i+n]
        result.append(batch)
    return result


def compute_embeddings(input_text, batch_size=8):
    text_batches = make_batches(input_text, batch_size)

    all_embeddings = []
    batch_idx = 0
    for batch in text_batches:
        print(f"Processing batch {batch_idx} of {len(text_batches)}")
        encoded_input = tokeniser(batch, padding=True, truncation=True,
                                  return_tensors='pt')

        with torch.no_grad():
            outputs = model(**encoded_input)
            hidden_states = outputs.last_hidden_state

            batch_embeddings = hidden_states.mean(dim=1)
            batch_embeddings_arr = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_arr)

        batch_idx += 1

    final_embeddings = np.vstack(all_embeddings)

    return final_embeddings


X_text = compute_embeddings(texts)
