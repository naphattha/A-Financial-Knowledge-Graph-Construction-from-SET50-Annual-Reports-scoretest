import streamlit as st
from llm import llm, embeddings
from db import database

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class CustomEmbeddingWrapper:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, text):
        # Use the encode method and ensure the output is a NumPy array
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed(self, text):
        return self.embed_query(text)

# Define the custom embedding model
custom_embeddings = CustomEmbeddingWrapper('sentence-transformers/all-MiniLM-L6-v2')


# Initialize Neo4jVector with the provided embeddings and graph
neo4jvector = Neo4jVector.from_existing_index(
    custom_embeddings,  # Use the correct embedding model
    graph=database,
    index_name="financialStatementVectorIndex",  # Ensure this is the correct index
    node_label="financial_statement",        # Ensure this is the correct node label
    text_node_property="financial_statement",                # Use 'uri' as the text property
    embedding_node_property="plotEmbedding", # Adjust if needed
    retrieval_query="""
    RETURN
        node.uri AS text,
        score,
        {
            ebitAccum: node.ebitAccum,
            netProfitAccum: node.netProfitAccum,
            totalRevenueAccum: node.totalRevenueAccum,
            totalAssets: node.totalAssets
        } AS metadata
    """
    # เปลี่ยนเป็นหาข้อมูล node industry
)



# Set up retriever
retriever = neo4jvector.as_retriever()

# Define the instructions and prompt for the LLM
instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "ให้คำตอบของออกมาเป็นภาษาไทยเป็นหลักโดยอาจมีคำศัพท์ทางการเงินบางอย่างที่เป็นภาษาอังกฤษได้"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create a question-answering chain using the LLM and prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create a retrieval chain that combines the retriever with the question-answer chain
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

def get_company_industry(input):
    # Use the retrieval chain to get the plot information
    return plot_retriever.invoke({"input": input})
