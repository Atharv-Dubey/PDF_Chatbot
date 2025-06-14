
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
from google_web_scrapper_bs4 import *
from tavily_data_generator import *

import logging
logging.disable(logging.WARNING)
#C:\Users\athar\PycharmProjects\WebScrapping\.venv\countries_info.pdf
import PyPDF2
import gc
import numpy as np
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import time

chunks = []
nlp = spacy.load("en_core_web_sm")
def read_and_extract_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please provide a valid PDF path.")
        return None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None



def build_rag_model(text):

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    chunks.extend(splitter.split_text(text))

    gc.collect()


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    batch_size = 100
    vector_store = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_texts(batch, embeddings)
        else:
            temp_store = FAISS.from_texts(batch, embeddings)
            vector_store.merge_from(temp_store)
            del temp_store  # Free memory
        gc.collect()
    return vector_store, embeddings


def add_new_text_to_model(new_text, vector_store, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    new_chunks = splitter.split_text(new_text)
    #chunks.extend(new_chunks)

    batch_size = 100
    temp_store = None
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        if temp_store is None:
            temp_store = FAISS.from_texts(batch, embeddings)
        else:
            batch_store = FAISS.from_texts(batch, embeddings)
            temp_store.merge_from(batch_store)
            del batch_store
        gc.collect()

    vector_store.merge_from(temp_store)
    del temp_store
    gc.collect()
    return vector_store



def setup_rag_pipeline(vector_store):
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0 if torch.cuda.is_available() else -1,
        max_length=150
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )



def extract_key_terms(query):
    doc = nlp(query.lower())

    key_terms = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] or token.ent_type_:
            key_terms.append(token.text)

    return list(dict.fromkeys(key_terms))


def get_significant_terms(chunks, top_n=50):

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(chunks)
    feature_names = vectorizer.get_feature_names_out()

    avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)

    top_indices = avg_scores.argsort()[-top_n:][::-1]
    significant_terms = [feature_names[idx] for idx in top_indices]
    return set(significant_terms)

def can_confidently_answer(query, chunks, threshold=0.7):

    key_terms = extract_key_terms(query)
    if not key_terms:
        print(0)
        return False

    pdf_terms = get_significant_terms(chunks)


    full_text = " ".join(chunks).lower()
    term_presence = sum(1 for term in key_terms if term in full_text)
    term_presence_ratio = term_presence / len(key_terms)


    context_overlap = sum(1 for term in key_terms if term in pdf_terms)
    context_overlap_ratio = context_overlap / len(key_terms) if key_terms else 0.0

    confidence_score = (0.6 * term_presence_ratio) + (0.4 * context_overlap_ratio)


    print(confidence_score)
    can_answer = confidence_score >= threshold
    return can_answer

def run_chatbot():

    file_path = input("Please enter the path to your PDF file (e.g., document.pdf): ")
    pdf_text = read_and_extract_pdf(file_path)
    if pdf_text is None:
        print("Exiting due to error in PDF processing.")
        return

    print(f" {file_path} extracted successfully.")

    vector_store, embeddings = build_rag_model(pdf_text)
    print("RAG model prepared successfully.")


    qa_chain = setup_rag_pipeline(vector_store)



    print("Web Integrated PDF-Chatbot is ready! Ask questions about the PDF content (type 'exit' to stop).")
    #print("1")
    while True:
        #print("2")
        time.sleep(2)
        query = input("Your question: ")
        print(query)
        if query.lower() == "exit":
            print("Thank You For Using the Web Integrated PDF Chatbot")
            break
        if not query.strip():
            print("Please enter a valid question.")
            continue
        #print("3")
        result = qa_chain({"query": query})
        answer = result["result"]
        print(f"Answer: {answer}\n")

        if can_confidently_answer(query, chunks)  == False :
            print("But I am not confident about this Answer, based on the Input PDF. I am going to use the web, Just a second.")
            new_text = answer_query(query)
            #new_text = get_answer_tavily(query)
            #print("4")
            if new_text == 0:
                print("Couldn't find any better answer")
            else:
                #print("5")
                vector_store = add_new_text_to_model(new_text, vector_store, embeddings)
                vector_store = add_new_text_to_model(new_text, vector_store, embeddings)
                qa_chain = setup_rag_pipeline(vector_store)
                print("Voila")
                result = qa_chain({"query": query})
                answer = result["result"]
                print(f"Answer: {answer}\n")






# Run the chatbot
#if __name__ == "__main__":
#    run_chatbot()