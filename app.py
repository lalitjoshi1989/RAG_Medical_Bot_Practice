from flask import Flask, render_template, jsonify, request
from src.helper import download_embed_model
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import requests
from src.connect_sql import connection_db


app = Flask(__name__)
load_dotenv()
conn, cursor = connection_db()

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

intent_prompt = PromptTemplate.from_template("""
You are an intent classifier. Classify the following user input into one of these categories:
- weather
- medical
- user

User input: "{query}"
Respond with only one word: weather, medical, or user.
""")


My_Api_Key = os.environ.get('My_Api_Key')

intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods = ['GET', 'POST'])
def chat():
    msg = request.form["msg"]
    intent = intent_chain.invoke({"query": msg})['text'].strip().lower()
    print("Current iNTENT IS", intent)

    if intent == 'weather':
        city = msg.strip().split()[-1]
        url1 = f"https://api.weatherapi.com/v1/current.json?key={My_Api_Key}&q={city}&aqi=no"
        r = requests.get(url1)
        data = r.json()
        try:
            temp = data['current']['temp_c']
            return str(f"The current temperature in {city} is {temp}°C.")
        except KeyError:
            return "Sorry, I couldn't find the weather for that location. Please try again with a valid city name."
    elif intent == 'user':        
        return jsonify({
        "status": "collect_info",
        "message": "Please provide your name, email, and phone number."
        })
    else:
        response = rag_chain.invoke({"input":msg})
        return str(response['answer'])

@app.route("/save_user", methods=["POST"])
def save_user():
    name = request.form.get("name")
    email = request.form.get("email")
    phone = request.form.get("phone")

    cursor.execute("CREATE TABLE IF NOT EXISTS user_info (Personid INT NOT NULL AUTO_INCREMENT, name VARCHAR(500), email VARCHAR(100), phone VARCHAR(100), PRIMARY KEY (Personid))")
    cursor.execute("INSERT INTO user_info (name, email, phone) VALUES (%s, %s, %s)", (name, email, phone))
    conn.commit()

    return "User information saved successfully."

if __name__ == "__main__":
    app.run(port="1181", debug=True)