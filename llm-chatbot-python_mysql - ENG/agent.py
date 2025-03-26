from llm import llm
from llm import embeddings
from db import database
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
# from tools.vector import get_company_industry
from tools.mysql_qa import mysql_qa_function
from tools.analysis import analysis_function
from tools.comparisons import comparisons_function
from tools.financial_statements import financial_statements_function
from tools.market_prices import market_prices_function

import streamlit as st
from langchain_core.prompts import PromptTemplate

# Initialize the LLM and embedding models
embed_model = embeddings

financial_statements_Tool = Tool.from_function(
    name="Financial Statements",
    description="Retrieve company financial figures such as revenue, profit, assets, and liabilities.",
    func=financial_statements_function
)

market_prices_Tool = Tool.from_function(
    name="Market Prices & Info",
    description="Fetch stock prices, P/E ratios, and other market-related data.",
    func=market_prices_function
)

comparisons_Tool = Tool.from_function(
    name="Comparisons",
    description="Compare financial ratios, trends, and figures across companies or time periods.",
    func=comparisons_function
)

analysis_Tool = Tool.from_function(
    name="Financial Analysis",
    description="Perform deeper financial analysis based on historical data and trends.",
    func=analysis_function
)

mysql_qa_Tool = Tool.from_function(
    name="search company's data",
    description="Use to find company's financial ratios and financial information using queries.",
    func=mysql_qa_function
)


tools = [
    financial_statements_Tool,
    market_prices_Tool,
    comparisons_Tool,
    analysis_Tool,
    mysql_qa_Tool
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

Use them when necessary. Follow this format:

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

import time
import streamlit as st

def generate_response(user_input):
    """
    Generates a response from the agent and includes metadata.
    """
    try:
        print(f"Raw user input: {user_input}")
        
        # Call mysql_qa_function to get query execution details
        df, query, query_gen_time, db_fetch_time, error = mysql_qa_function(user_input)

        # Start measuring response generation time
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

        return final_response, metadata, None
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



# def generate_response(user_input):
#     """
#     Generates a response from the agent and includes metadata.
#     Collects response generation time and integrates query execution details.
#     """
#     try:
#         print(f"Raw user input: {user_input}")
        
#         # Start measuring response generation time
#         start_response_time = time.time()

#         # Call the mysql_qa_function
#         df, query, query_gen_time, db_fetch_time, error = mysql_qa_function(user_input)

#         # End measuring response generation time
#         end_response_time = time.time()

#         # Calculate response generation time
#         response_generation_time = end_response_time - start_response_time

#         # Combine metadata
#         metadata = {
#             "query": query,
#             "query_generation_time": query_gen_time,
#             "database_fetch_time": db_fetch_time,
#             "response_generation_time": response_generation_time,
#             "error": error,
#         }

#         # If error occurred, return metadata with no response
#         if error:
#             return None, metadata, error

#         # Final response
#         final_response = f"Query results:\n{df.to_string(index=False)}" if not df.empty else "No data found."
        
#         return final_response, metadata, None
#     except Exception as e:
#         print(f"Error in generate_response: {e}")
#         return None, {}, str(e)

