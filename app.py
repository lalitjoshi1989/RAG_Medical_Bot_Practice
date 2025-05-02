from flask import Flask, render_template, jsonify, request
from src.helper import download_embed_model
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

pinecone_api_key = os.environ.get('pinecone_api_key')
os.environ["pinecone_api_key"] = pinecone_api_key

embeddings = download_embed_model()

index_name = 'medicalbot'

docsearch = PineconeVectorStore.from_existing_index(
    index_name= index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type = "similarity",
                       search_kwargs = {"k":3})


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=50
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods = ['GET', 'POST'])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input":msg})
    return str(response['answer'])

if __name__ == "__main__":
    app.run(port="1181", debug=True)