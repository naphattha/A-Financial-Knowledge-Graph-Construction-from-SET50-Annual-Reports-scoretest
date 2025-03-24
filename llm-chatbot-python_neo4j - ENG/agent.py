from llm import llm
from llm import embeddings
from fkg import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
# from tools.vector import get_company_industry
from tools.cypher_qa import cypher_qa_function

import streamlit as st
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex, load_index_from_storage
from langchain_groq import ChatGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.prompts import PromptTemplate
from utils import get_session_id

# Initialize the LLM and embedding models
embed_model = embeddings

from langchain.schema import SystemMessage, HumanMessage


def simple_function(input_text):
    return f"Received input: {input_text}"

General_Chat = Tool.from_function(
    name="General Chat",
    description="For general conversations about financial information not covered by other tools",
    func=simple_function
)

# company_industry_Tool = Tool.from_function(
#     name="company industry Search",
#     description="Use to find out which industry a company belongs to",
#     func=get_company_industry
# )


cypher_qa_Tool = Tool.from_function(
    name="search company's data",
    description="Use to find company's financial ratios and financial information using queries.",
    func=cypher_qa_function
)

tools = [
    General_Chat,
    cypher_qa_Tool
]

# company_industry_Tool,

prompt_template = PromptTemplate.from_template("""
You are a financial expert tasked with providing accurate and comprehensive information and advice related to financial matters. This includes company data, investments, market conditions, and economic trends.

Language Instructions:
- Provide answers primarily in English.
- Ensure clarity and precision in financial terminology.

Source of Information:
- Use only the information available in the context provided.
- Do not use knowledge learned independently.

TOOLS:
You can use the following tools, but avoid using General_Chat if possible:

{tools}

To use a tool, please follow these steps:

1. Determine if the information needed to answer the question is available in the context.
2. If the information is not available, decide if the tool is appropriate for retrieving it.
3. If the tool should be used, follow this format:

```
Thought: Is it necessary to use a tool? Yes
Action: The action to be taken should be one of [{tool_names}]
Action Input: Information used for the action
Observation: The result of the action
```

When you have an answer to provide to the user or if it's unnecessary to use a tool, use the following format:
                                               
```
Thought: Is it necessary to use a tool? No
Final Answer: [Your answer here]
```

Begin!

Previous conversation history:
{chat_history}

New message: {input}
{agent_scratchpad}
""")


# Create the agent instance
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

# Setup the agent executor with error handling
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    prompt=prompt_template,
    verbose=True,
    handle_parsing_errors=True  # Enable error handling for parsing issues
)

import time  # Import the time module

# Function to generate response
def generate_response(user_input):
    """
    Generates a response from the agent and includes metadata.
    """
    try:
        print(f"Raw user input: {user_input}")
        
        output= cypher_qa_function(user_input)

        data = output.get('data')
        query = output.get('query')  # Extract the query
        query_gen_time = output.get('query_generation_time')  # Extract query generation time
        db_fetch_time = output.get('database_fetch_time')  # Extract database fetch time
        error = output.get('error')  # Extract error, if any

        if error:
            raise ValueError(f"Error in cypher query generation: {error}")
        
        start_time = time.time()

        # Pass query execution details to the agent
        agent_response = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.get("chat_history", []),
            "query": query,
            "query_generation_time": query_gen_time,
            "database_fetch_time": db_fetch_time,
            "sql_error": error,
        })

        # End measuring response generation time
        end_time = time.time()
        response_generation_time = end_time - start_time

        # Debug: Print agent response
        print(f"Agent response: {agent_response}")
        print(f"Response generation time: {response_generation_time:.5f} seconds")

        # Extract the response and metadata
        final_response = agent_response.get("output", "No output found")

        # Construct metadata
        metadata = {
            "query": query,
            "response": final_response,
            "query_generation_time": query_gen_time,
            "database_fetch_time": db_fetch_time,
            "response_generation_time": response_generation_time,
            "error": error,
        }

        return metadata, None
    except Exception as e:
        print(f"Error generating response: {e}")
        metadata = {
            "query": None,
            "response": "Error generating response",
            "query_generation_time": None,
            "database_fetch_time": None,
            "response_generation_time": None,
            "error": str(e),
        }
        return None, metadata, f"Error: Unable to generate response due to {str(e)}"