# %% load the libraries
from openai import OpenAI
from elasticsearch import Elasticsearch
import requests
from tqdm.auto import tqdm


# %% define the LLM client
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama'
)


# %% create the elastic search index
es = Elasticsearch("http://localhost:9200")

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
            "course": {"type": "keyword"}
        }
    }
}

index_name = "course-questions"
response = es.indices.create(index=index_name, body=index_settings)


# %% load and parse the data
docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)


# %% index the documents
for doc in tqdm(documents):
    es.index(index=index_name, document=doc)


# %% define a search function
def retrieve_documents(query):   
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    
    response = es.search(index=index_name, body=search_query)
    
    result_docs = [hit['_source'] for hit in response['hits']['hits']]
    return result_docs


# %% define the prompt functions
def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT: {context}
    """.strip()

    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt):
    response = client.chat.completions.create(
        model='phi3',
        messages=[{"role": "user", "content": prompt}]
    )    
    return response.choices[0].message.content


# %% define RAG function
def rag(query):
    search_results = retrieve_documents(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# %% test the operation of the rag function
query = "I just discovered the course. Can I still join it?"
print(rag(query))
