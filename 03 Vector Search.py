## import libraries
import json
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

## step 1: prepare the documents
with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)

# flatten the json file (required for elasticsearch)
documents = []
for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

sample_doc = documents[2]
print(sample_doc)

## step 2: create the embeddings using a pre-trained model
# load the embedding model
embed_model = SentenceTransformer("all-mpnet-base-v2")

# demonstrate a simple example
# return a dense vector
dv = embed_model.encode(sample_doc['text'])
print(dv)
# determine the length of that vector
print(len(dv))

# create dense vectors for all the 'text' fields
# this takes a couple of minutes
dvs = []
for doc in documents:
    doc['text_vector'] = embed_model.encode(doc['text']).tolist()
    dvs.append(doc)

print(dvs[:1])

## step 3: set up elastic search connection
# check that the connection has been established
es_client = Elasticsearch('http://localhost:9200')
print(es_client.info())

## step 4: create the mappings and index
# mapping is the process of defining how a document, and the fields it contains, are stored and indexed
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            # this is a new field; the dimension can be found on the model card; also specify the similarity metric used
            "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}

index_name = "course-questions"
# delete any index with the same name
es_client.indices.delete(index=index_name, ignore_unavailable=True)
# create the index
es_client.indices.create(index=index_name, body=index_settings)

## step 5: add documents to the index
# again, this step can take a couple of minutes
for doc in dvs:
    try:
        es_client.index(index=index_name, document=doc)
    except Exception as e:
        print(e)

## step 6: create the end-user query
user_query = "Should I use Windows or Mac?"
vectorised_user_query = embed_model.encode(user_query)

search_query = {
    # what field should be searched
    "field": "text_vector",
    # what's the input query
    "query_vector": vectorised_user_query,
    # how many results to return
    "k": 5,
    "num_candidates": 10000
}

# pass the search_query dict to elasticsearch
# source indicates what (metadata) fields should be returned
response = es_client.search(index=index_name, knn=search_query)
, fields={"text", "section", "question", "course"})

## step 7: perform semantic search

