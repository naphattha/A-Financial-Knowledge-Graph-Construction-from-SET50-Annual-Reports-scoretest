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
You are an expert in MySQL who translates user questions into SQL queries to find answers about company data in the database.
Translate the user's questions according to the database schema.
Use only the types of relationships and properties that exist in the given schema. Do not use other types of relationships or properties that are not in the provided schema.

Fine Tuning:
1.FilteredEODData: table for company financial statement data, available quarterly (de, ebitAccum, ebitQuarter, epsAccum, epsQuarter, financingCashFlow, fixedAssetTurnover, hasQuarter, hasYear, investingCashFlow, netProfitAccum, netProfitMarginAccum, netProfitMarginQuarter, netProfitQuarter, operatingCashFlow, paidupShareCapital, roa, roe, shareholderEquity, totalAssetTurnover, totalAssets, totalEquity, totalExpensesAccum, totalExpensesQuarter, totalLiabilities, totalRevenueAccum, totalRevenueQuarter)
    financial_statements: tablefor daily company stock price data (average, close, high, low, open, prior, totalVolume), with the date information in the URI, e.g., uri: http://example.org/stock_value_CPALL_2024-01-25.
2. The SQL query must be written on a single line without any line breaks. Avoid using `\n` or multi-line formatting.
3.Example Questions and Translations:
    3.1. คำถาม: บริษัท ADVANC มีสินทรัพย์รวมในไตรมาสที่ 1 ปี 2019 เท่าไหร่
        SQL Query: `SELECT totalAssets FROM financial_statements WHERE symbol = 'ADVANC' AND year = 2019 AND quarter = 1 LIMIT 1;`
    3.2. คำถาม: บริษัท AOT มีหนี้สินรวมในไตรมาสที่ 1 ปี 2019 เท่าไหร่
        SQL Query: `SELECT totalLiabilities FROM financial_statements WHERE symbol = 'AOT' AND year = 2019 AND quarter = 1 LIMIT 1;`
    3.3. คำถาม: กำไรสุทธิของบริษัท BBL ในไตรมาสที่ 1 ปี 2019 คือเท่าไหร่
        SQL Query: `SELECT netProfitQuarter FROM financial_statements WHERE symbol = 'BBL' AND year = 2019 AND quarter = 1 LIMIT 1;`
    3.4. คำถาม: อัตราผลตอบแทนต่อผู้ถือหุ้น (ROE) ของบริษัท BCP ในไตรมาสที่ 1 ปี 2019 คือเท่าไหร่
        SQL Query: `SELECT roe FROM financial_statements WHERE symbol = 'BCP' AND year = 2019 AND quarter = 1 LIMIT 1;`
    3.5. คำถาม: สัดส่วนหนี้สินต่อทุน (D/E) ของบริษัท BDMS ในไตรมาสที่ 1 ปี 2019 คือเท่าไหร่
        SQL Query: `SELECT de FROM financial_statements WHERE symbol = 'BDMS' AND year = 2019 AND quarter = 1 LIMIT 1;`
    3.6. คำถาม: รายได้รวมของบริษัท BEM ในไตรมาสที่ 1 ปี 2019 เท่าไหร่
        SQL Query: `SELECT totalRevenueQuarter FROM financial_statements WHERE symbol = 'BEM' AND year = 2019 AND quarter = 1 LIMIT 1;`
    3.7. คำถาม: เปรียบเทียบสินทรัพย์รวมของบริษัท ADVANC กับบริษัท AOT ในปี 2019
        SQL Query: `SELECT symbol, Quarter, totalAssets FROM financial_statements WHERE symbol IN ('ADVANC', 'AOT') AND year = 2019;`
    3.8. คำถาม: เปรียบเทียบหนี้สินรวมของบริษัท ADVANC กับบริษัท BBL ในปี 2019
        SQL Query: `SELECT symbol, Quarter, totalLiabilities FROM financial_statements WHERE symbol IN ('ADVANC', 'BBL') AND year = 2019;`
    3.9. คำถาม: เปรียบเทียบรายได้รวมในไตรมาสของบริษัท ADVANC กับบริษัท CPALL ในไตรมาสที่ 3 ปี 2021
        SQL Query: `SELECT symbol, totalRevenueQuarter FROM financial_statements WHERE symbol IN ('ADVANC', 'CPALL') AND year = 2021 AND quarter = 3;`
    3.10. คำถาม: ราคาปิดของหุ้น AWC ในวันที่ 1 กันยายน 2023 คือเท่าไหร่
        SQL Query: `SELECT close FROM filteredEODData WHERE symbol = 'AWC' AND date = '2023-09-01' LIMIT 1;`

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

