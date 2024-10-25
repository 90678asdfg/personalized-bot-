import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.question_answering import load_qa_chain




# Constants for the model and server URL
MODEL = "llama3"
OLLAMA_SERVER_URL = "http://45.114.48.221:11434/"  # Local server URL
FILE_PATH = "faiss_store_ollama.pkl"

# Initialize the model and embeddings
model = Ollama(model=MODEL, base_url=OLLAMA_SERVER_URL)
embeddings = OllamaEmbeddings(model=MODEL, base_url=OLLAMA_SERVER_URL)

st.title("Your Personalized Info Insight Tool")
st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", value="")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

def fetch_data_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return None

if process_url_clicked:
    data = []
    for url in urls:
        if url:
            fetched_data = fetch_data_from_url(url)
            if fetched_data:
                data.append(fetched_data)
    
    if not data:
        st.error("No data loaded from the provided URLs. Please check the URLs and try again.")
        raise ValueError("No data loaded from the provided URLs.")
    
    # Process data directly without text_splitter
    docs = [{"content": doc} for doc in data]  # Format each document as {"content": text}

    if not docs:
        st.error("No documents created from the loaded data. Please check the URLs and content.")
        raise ValueError("No documents created from the loaded data.")
    
    # Create embeddings and save them to the FAISS index
    vectorstore_ollama = FAISS.from_texts([doc["content"] for doc in docs], embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(FILE_PATH, "wb") as f:
        pickle.dump(vectorstore_ollama, f)

# Prompt template for question answering
template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

# Load the QA chain
qa_chain = load_qa_chain(llm=model, chain_type="stuff")

def query_rag_system(question):
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "rb") as f:
            vectorstore = pickle.load(f)
        retriever = vectorstore.as_retriever(k=4)

        # Combine the components into a RetrievalQA chain
        qa_retrieval_chain = RetrievalQA(
            retriever=retriever,
            combine_documents_chain=qa_chain,
            input_key="question",
            output_key="answer"
        )

        response = qa_retrieval_chain({"question": question})
        return response["answer"]
    else:
        st.error("FAISS vector store not found. Please process the URLs first.")
        return None

query = main_placeholder.text_input("Question: ")
if query and st.button("Get Answer"):
    answer = query_rag_system(query)
    if answer:
        st.header("Answer")
        st.write(answer)
    else:
        st.write("Could not retrieve an answer. Please check the inputs and try again.")
