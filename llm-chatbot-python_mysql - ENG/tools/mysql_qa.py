import streamlit as st
from llm import llm
from db import database, db_url
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from sqlalchemy import create_engine, inspect
import pandas as pd

# Construct the database URL
MYSQL_HOST = st.secrets["MYSQL_HOST"]
MYSQL_USER = st.secrets["MYSQL_USER"]
MYSQL_PASSWORD = st.secrets["MYSQL_PASSWORD"]
MYSQL_DB = st.secrets["MYSQL_DB"]


db_url = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}"

# Create the SQLAlchemy engine
engine = create_engine(db_url)

# Create an instance of SQLDatabase using the SQLAlchemy engine
database = SQLDatabase(engine)

MYSQL_GENERATION_TEMPLATE = """
You are an expert in MySQL who translates user questions into SQL queries to retrieve company data from the database. 
Translate the user's question according to the provided database schema and strictly adhere to these rules:

Fine Tuning:
1. **Schema Details**:
    - 'Company': Table for company details, including attributes such as symbol, name.
    - 'Period': Table for financial period data, including attributes such as year, quarter, date.
    - 'FinancialStatements': Table for company financial data by quarter, including attributes such as symbol, year, quarter, totalAssets, totalLiabilities, totalRevenueQuarter, netProfitQuarter, etc.
    - 'FinancialRatios': Table for calculated financial ratios, including attributes such as roe, roa, netProfitMarginQuarter, de.
    - 'MarketRatios': Table for market-related ratios, including attributes such as pe, pbv, dividendYield, marketCap.
    - 'StockPrices': Table for daily company stock price data, including attributes such as symbol, date, open, high, low, close, volume, totalValue.

2. **Output Rules**:
   - Write SQL queries as a single line without line breaks or extra text.
   - Do not include additional explanations or preamble.
   - If the user's question cannot be answered with the schema provided, respond only with: `I don't know`.

3. **Example Questions and Queries**:
    - Question: What was the total assets of ADVANC in Q1 2019?
      SQL Query: `SELECT totalAssets FROM financial_statements WHERE symbol = 'ADVANC' AND year = 2019 AND quarter = 1 LIMIT 1;`
    - Question: Compare the total liabilities of ADVANC and BBL in 2019.
      SQL Query: `SELECT symbol, quarter, totalLiabilities FROM financial_statements WHERE symbol IN ('ADVANC', 'BBL') AND year = 2019;`
    - Question: What was the closing price of AWC stock on September 1, 2023?
      SQL Query: `SELECT close FROM FilteredEODData WHERE symbol = 'AWC' AND date = '2023-09-01' LIMIT 1;`

Schema:
{schema}

Question:
{question}

Mysql Query:
"""

# Define the tools for SQL generation and execution
execute_query = QuerySQLDataBaseTool(db=database)
write_query = create_sql_query_chain(llm, database)
# Create the prompt template
sql_prompt = PromptTemplate.from_template(MYSQL_GENERATION_TEMPLATE)

# Define the LangChain QA chain
mysql_qa = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | sql_prompt
    | llm
    | StrOutputParser()
)

# Function to fetch schema
def get_schema(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    schema = {}
    for table in tables:
        columns = inspector.get_columns(table)
        schema[table] = {col['name']: str(col['type']) for col in columns}
    return schema

# Convert schema to string format for prompt
def format_schema(schema):
    formatted = ""
    for table, columns in schema.items():
        formatted += f"Table: {table}\n"
        for column, type_ in columns.items():
            formatted += f"  - Column: {column}, Type: {type_}\n"
    return formatted

# Fetch schema from the database
schema1 = get_schema(engine)


import time
import pandas as pd
from sqlalchemy import text

def mysql_qa_function(input_text):
    """
    Executes an SQL query generated for the input text and logs its execution details.
    Collects the following times:
    - Time taken to generate the query.
    - Time taken to execute the query (database drag time).
    """
    try:
        logs = []

        # Measure query generation time
        start_query_gen_time = time.time()

        # Fetch schema from the database
        schema_str = format_schema(schema1)
        # Generate SQL query specific to the user's input question
        generated_result = mysql_qa.invoke({"question": input_text, "schema": schema_str})
        end_query_gen_time = time.time()

        # print('asdasdasdadc:',generated_result)
        
        # # Extract the query from the generated result
        # query_start = generated_result.find("```\n")
        # query_end = generated_result.find("\n```", query_start)
        # query = generated_result[query_start:query_end].strip()

        # Ensure the query is in one line
        query = " ".join(generated_result.split())

        # If query is empty or invalid, return an empty dataframe
        if not query:
            return pd.DataFrame(), query, end_query_gen_time - start_query_gen_time, 0.0, None


        # Measure database drag time
        start_db_time = time.time()

        with engine.connect() as connection:
            connection.execute(text("USE financials;"))
            result = connection.execute(text(query))
            data = result.fetchall()

        end_db_time = time.time()

        # Convert results to a pandas DataFrame
        df = pd.DataFrame(data, columns=result.keys())

        return (
            df,
            query,
            end_query_gen_time - start_query_gen_time,
            end_db_time - start_db_time,
            None,  # No error
        )
    except Exception as e:
        print(f"Error executing query: {e}")
        st.session_state.query_logs.append({
            "query": query if 'query' in locals() else None,
            "query_generation_time": end_query_gen_time - start_query_gen_time if 'end_query_gen_time' in locals() else 0.0,
            "database_fetch_time": end_db_time - start_db_time if 'end_db_time' in locals() else 0.0,
            "error": str(e),
        })
        return (
            pd.DataFrame(),
            query if 'query' in locals() else None,
            end_query_gen_time - start_query_gen_time if 'end_query_gen_time' in locals() else 0.0,
            end_db_time - start_db_time if 'end_db_time' in locals() else 0.0,
            str(e),  # Error message
        )

# Export the tool
__all__ = ["mysql_qa_function"]

