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
    - 'company': Table for company details, including attributes such as id, symbol, name.
    - 'period': Table for financial period data, including attributes such as id, year, quarter, date.
    - 'financialmetrics': financialmetrics: Table for company financial data by quarter, including attributes such as id, company_id, period_id, total_assets, total_liabilities, total_revenue_quarter, net_profit_quarter, etc.
    - 'financialratios': Table for calculated financial ratios, including attributes such as id, company_id, period_id, and types like ROE, ROA, NetProfitMarginQuarter, NetProfitMarginAccum, DE, FixedAssetTurnover, TotalAssetTurnover.
    - 'marketratios': Table for market-related ratios, including attributes such as id, company_id, period_id, and types like PE, PBV, BVPS, DividendYield, MarketCap, VolumeTurnover.
    - 'marketdata': Table for daily company stock price data, including attributes such as id, company_id, period_id, open, high, low, close, volume, total_value.

2. **Output Rules**:
   - Write SQL queries as a single line without line breaks or extra text.
   - Do not include additional explanations or preamble.
   - Do not add any text before or after the SQL query. Only output the SQL query.

3. **Example Questions and Queries**:
    - Question: อัตราการหมุนเวียนของสินทรัพย์ถาวรของ BCP ส่งผลต่อกำไรจากการดำเนินงานอย่างไร
      SQL Query: SELECT p.year, p.quarter, r.value AS fixed_asset_turnover, f.operating_cash_flow FROM financialmetrics f JOIN financialratios r ON f.company_id = r.company_id AND f.period_id = r.period_id JOIN company c ON f.company_id = c.id JOIN period p ON f.period_id = p.id WHERE c.symbol = 'BCP' AND r.type = 'fixedAssetTurnover';
    - Question: แนวโน้มของกระแสเงินสดจากการดำเนินงานของ BGRIM ระหว่างปี 2019-2021 เป็นอย่างไร
      SQL Query: SELECT p.year, p.quarter, f.operating_cash_flow FROM financialmetrics f JOIN company c ON f.company_id = c.id JOIN period p ON f.period_id = p.id WHERE c.symbol = 'BGRIM' AND p.year BETWEEN 2019 AND 2021 ORDER BY p.year, p.quarter;
    - Question: อัตราส่วนหนี้สินต่อทุน (DE) ของ ADVANC มีการเปลี่ยนแปลงอย่างไรในแต่ละไตรมาสของปี 2019
      SQL Query: SELECT p.quarter, r.value AS de_ratio FROM financialratios r JOIN company c ON r.company_id = c.id JOIN period p ON r.period_id = p.id WHERE c.symbol = 'ADVANC' AND r.type = 'DE' AND p.year = 2019 ORDER BY p.quarter;


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

def analysis_function(input_text):
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
            connection.execute(text("USE set50;"))  # edit this to the name of database
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
__all__ = ["analysis_function"]

