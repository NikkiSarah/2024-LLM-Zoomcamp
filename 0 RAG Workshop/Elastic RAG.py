#%% Load and Prepare Data
import os
import json

# check the current working directory
os.getcwd()

# unnest/flatten the file structure
with open ('documents.json', 'rt') as f:
    data_file = json.load(f)

documents = []
for course in data_file:
    course_name = course['course']
    
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

print(documents[2])
print(f"There are: {len(documents)} documents.")

 
# %% Index Data with elasticsearch
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

# check it's working
es = Elasticsearch("http://localhost:9200")

print(es.info())

# create an index (this is like a table in a relational db)
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
response = es.indices.create(index=index_name, body=index_settings)

print(response)

# index all the documents
for doc in tqdm(documents):
    es.index(index=index_name, document=doc)


# %% Retrieve the Results of a Search
user_question = "How do I join the course after it has started?"

search_query = {
    "size": 5,  # retrieve the top-5 matching documents
    "query": {
        "bool": {
            "must": {
                # search over the specified fields, prioritising "question"
                "multi_match": {
                    "query": user_question,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            # search only the data engineering course documents
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}

# perform the search
response = es.search(index=index_name, body=search_query)
print(response)

# view a "prettier" version of the results
for hit in response['hits']['hits']:
    doc = hit['_source']
    print(f"Section: {doc['section']}")
    print(f"Question: {doc['question']}")
    print(f"Answer: {doc['text'][:60]}...\n")


# %% Document Retrieval Function
es = Elasticsearch("http://localhost:9200")

def retrieve_documents(query, index_name="course-questions", max_results=5):   
    search_query = {
        "size": max_results,
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
    documents = [hit['_source'] for hit in response['hits']['hits']]
    return documents


# check the function works
user_question = "How do I run kafka?"

response = retrieve_documents(user_question)

for doc in response:
    print(f"Section: {doc['section']}")
    print(f"Question: {doc['question']}")
    print(f"Answer: {doc['text'][:60]}...\n")


# %% Communicate with an Open-Source LLM
from cohere.client import Client

# co_api_key = os.environ['CO_API_KEY']
# co_client = Client(co_api_key)
co_client = Client()

response = co_client.chat(
    message="How do I get better at data science?",
    model="command-r",
    preamble=""
    )
print(response.text)


# %% Create a Prompt (Template)

# construct the context template
context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()

# retrieve all the results for the query
context_docs = retrieve_documents(user_question)

context_result = ""

# for each of the results...
for doc in context_docs:
    # concatenate the free-text fields into a single string
    doc_str = context_template.format(**doc)
    context_result += ("\n\n" + doc_str)

context = context_result.strip()
print(context)

# build the prompt that will be passed to the open-source LLM
# note that the "NONE" section is important as it otherwise will attempt to answer the
# question even if there is no relevant content
prompt = f"""
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database. 
Only use the facts from the CONTEXT.
If the CONTEXT doesn't contan the answer, return "NONE"

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()

# pass it to the Cohere API
response = co_client.chat(
    message=prompt,
    model="command-r",
    preamble=""
    )
print(response.text)


# %% RAG Function
context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()

prompt_template = """
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()


def build_context(documents):
    context_result = ""
    
    for doc in documents:
        doc_str = context_template.format(**doc)
        context_result += ("\n\n" + doc_str)
    
    return context_result.strip()


def build_prompt(user_question, documents):
    context = build_context(documents)
    prompt = prompt_template.format(
        user_question=user_question,
        context=context
    )
    return prompt


def ask_cohere(prompt, model="command-r"):
    response = co_client.chat(
        message=prompt,
        model="command-r",
        preamble=""
    )
    answer = response.text
    return answer


def qa_bot(user_question):
    context_docs = retrieve_documents(user_question)
    prompt = build_prompt(user_question, context_docs)
    answer = ask_cohere(prompt)
    return answer


# check the function works with different queries
qa_bot("I'm getting invalid reference format: repository name must be lowercase")
qa_bot("I can't connect to postgres port 5432, my password doesn't work")
qa_bot("how can I run kafka?")