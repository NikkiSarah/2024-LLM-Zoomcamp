import os
import json
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

os.getcwd()
es = Elasticsearch("http://localhost:9200")


def load_data(json_file_path):
    with open (json_file_path, 'rt') as f:
        data_file = json.load(f)

    documents = []
    for course in data_file:
        course_name = course['course']
        
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    return documents


def index_data(documents):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},  # the reply
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"}  # not a free-response field
            }
        }
    }

    index_name = "course-questions"
    try:
        response = es.indices.create(index=index_name, body=index_settings)

        for doc in tqdm(documents):
            es.index(index=index_name, document=doc)
    except:
        pass


def load_and_index_data(json_file_path):
    documents = load_data(json_file_path)
    index_data(documents)
    