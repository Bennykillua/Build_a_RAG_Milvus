FROM python:3.9-slim

WORKDIR /app

COPY . /app


RUN pip install -r requirements.txt

EXPOSE 8501

ENV OPENAI_API_KEY=your_openai_api_key
ENV MILVUS_ENDPOINT=./milvus_demo.db
ENV COLLECTION_NAME=my_rag_collection

CMD ["streamlit", "run", "build_rag_with_milvus.py"]
