# Environmental Research Assistant using RAG

This project implements a Retrieval-Augmented Generation (RAG) system for querying environmental and wildfire research papers.

## Features

• Upload environmental research papers  
• Semantic search using embeddings  
• FAISS vector database  
• LLM-based question answering  
• Streamlit user interface

## Tech Stack

Python  
LangChain  
FAISS  
OpenAI API  
Streamlit

## Architecture

1. Load environmental research PDF
2. Split document into text chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS vector database
5. Retrieve relevant chunks
6. Generate answers using LLM

## Run Locally

pip install -r requirements.txt

streamlit run app.py
