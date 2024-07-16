import os
from cohere.client import Client
from elasticsearch import Elasticsearch
import streamlit as st
from dataprep import load_and_index_data


load_and_index_data('documents.json')


# api_key = os.environ['CO_API_KEY']
# co_client = Client(co_api_key)
co_client = Client()

es = Elasticsearch("http://localhost:9200")
index_name = "course-questions"


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


def retrieve_documents(
        query,
        index_name="course-questions",
        max_results=5,
        course="data-engineering-zoomcamp"
    ):
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
                        "course": course
                    }
                }
            }
        }
    }
    
    response = es.search(index=index_name, body=search_query)
    documents = [hit['_source'] for hit in response['hits']['hits']]
    return documents


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


def qa_bot(user_question, course):
    context_docs = retrieve_documents(user_question, course=course)
    prompt = build_prompt(user_question, context_docs)
    answer = ask_cohere(prompt)
    return answer
