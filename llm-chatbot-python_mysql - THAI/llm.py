# llm.py

import os
import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_groq import ChatGroq

# Set the Groq API key securely
os.environ["LLAMA_API_KEY"] = st.secrets["LLAMA_API_KEY"]

# Define the LLM using Groq's Llama3
llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("LLAMA_API_KEY"))

# Define the embedding model
embeddings = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
