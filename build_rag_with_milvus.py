# -*- coding: utf-8 -*-
import os
import requests
import json
from tqdm import tqdm
from openai import OpenAI
from pymilvus import MilvusClient

# # Ensure your API Key is set up securely
os.environ["OPENAI_API_KEY"] = 'your_openai_api_key_here'

# # Initialize OpenAI and Milvus clients
openai_client = OpenAI()
milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"

# Initialize OpenAI and Milvus clients with hardcoded API key
# openai_api_key = ''
# openai_client = OpenAI(api_key=openai_api_key) 
# milvus_client = MilvusClient(uri="./milvus_demo.db")
# collection_name = "my_rag_collection"


def download_docs():
    api_url = "https://api.github.com/repos/milvus-io/milvus/contents/docs/developer_guides"
    raw_base_url = "https://raw.githubusercontent.com/milvus-io/milvus/master/docs/developer_guides/"
    docs_path = "milvus_docs"

    if not os.path.exists(docs_path):
        os.makedirs(docs_path)

    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file['name'].endswith('.md'):
                file_url = raw_base_url + file['name']
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    with open(os.path.join(docs_path, file['name']), "wb") as f:
                        f.write(file_response.content)
                    print(f"Downloaded: {file['name']}")
                else:
                    print(f"Failed to download: {file_url} (Status code: {file_response.status_code})")
    else:
        print(f"Failed to fetch file list from {api_url} (Status code: {response.status_code})")


def prepare_text_lines():
    from glob import glob
    text_lines = []
    for file_path in glob(os.path.join("milvus_docs", "*.md"), recursive=True):
        with open(file_path, "r", encoding="utf-8") as file:
            file_text = file.read()
            text_lines += file_text.split("# ")
    return text_lines


def emb_text(text):
    return openai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding


def create_collection():
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    test_embedding = emb_text("This is a test")
    embedding_dim = len(test_embedding)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",
        consistency_level="Strong",
    )


def insert_data(text_lines):
    data = []
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": emb_text(line), "text": line})
    milvus_client.insert(collection_name=collection_name, data=data)


def search_and_get_response(question):
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(question)],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    
    context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])
    
    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    
    return response.choices[0].message.content


def main():
    download_docs()
    text_lines = prepare_text_lines()
    create_collection()
    insert_data(text_lines)
    
    while True:
        question = input("Please enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = search_and_get_response(question)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
