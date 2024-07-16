# %% Introduction
from dotenv import load_dotenv, dotenv_values
import os
import requests
from cohere.client import Client

# load the api key
load_dotenv()

# check the environment contains the key
os.environ
## Cohere
client = Client()

response = client.chat(
    message="Is it too late to join the course?",
    model="command",
    preamble=""
    )
response.text

## Hugging Face
config = dotenv_values(".env")

model_url = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {config['HF_API_KEY']}"}
def query(payload):
    response = requests.post(model_url, headers=headers, json=payload)
    return response.json()
data = query({"inputs": "Is it too late to join the course? "})
print(data[0]['generated_text'])


# %% Retrieval and Search
# download the search engine script from the workshop
# import wget

# url = "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-intro/minsearch.py"
# file_path = "minsearch.py"
# wget.download(url, file_path)

import minsearch
import json

# # retrieve the FAQ data
# url = "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-intro/documents.json"
# file_path = "documents.json"
# wget.download(url, file_path)

with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)

# unnest the documents to get a list of dictionaries
documents = []
for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)
print(documents[0])

# now index the documents
index = minsearch.Index(
    text_fields=["question", "text", "section"],  # fields to search over
    keyword_fields=["course"]  # filtering fields
    )
index.fit(documents)

# perform the search
query = "The course has already started. Can I still enrol?"

boost = {'question': 3., 'section': 0.5}

search_results = index.search(
    query=query,
    filter_dict={'course': 'data-engineering-zoomcamp'},
    boost_dict=boost,
    num_results=5
    )
print(search_results)


# %% Generating Answers with a Pre-Trained LLM

q = "The course has already started. Can I still enrol?"

# load the api key and the client
load_dotenv()
client = Client()

response = client.chat(
    message=q,
    model="command",
    preamble=""
    )
response.text

# use a prompt template to get a better response
prompt_template = """
You're a course teaching assistant.
Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
If the CONTEXT doesn't contain the answer, output NONE

QUESTION: {question}
CONTEXT: {context}
""".strip()

context = ""
for doc in search_results:
    context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
print(context)

prompt = prompt_template.format(question=q, context=context).strip()
print(prompt)

response = client.chat(
    message=prompt,
    model="command",
    preamble=""
    )
response.text


# %% RAG Flow Cleaning and Code Modularisation

def search(query):
    boost = {'question': 3., 'section': 0.5}

    search_results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=10
        )
    return search_results


def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant.
    Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}
    CONTEXT: {context}
    """.strip()

    context = ""
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt


def call_llm(prompt):
    load_dotenv()
    client = Client()
    response = client.chat(
        message=prompt,
        model="command",
        preamble=""
        )
    return response


# check the functions work
query = "how do I run kafka?"
search_results = search(query)
prompt = build_prompt(query, search_results)
answer = call_llm(prompt)
print(answer.text)

query = "The course has already started. Can I still enrol?"
search_results = search(query)
prompt = build_prompt(query, search_results)
answer = call_llm(prompt)
print(answer.text)


# put everything into a simple RAG function
def RAG(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = call_llm(prompt)

    return answer.text


query = "Do I need to know Docker?"
RAG(query)


# %% Search with ElasticSearch



