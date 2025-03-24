import streamlit as st
from llm import llm
from fkg import graph
import pandas as pd
from neo4j import GraphDatabase
import time
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

# Function to fetch schema from Neo4j
def get_schema_from_neo4j(driver):
    with driver.session() as session:
        # Example: Get all labels
        labels_result = session.run("CALL db.labels()")
        labels = [record["label"] for record in labels_result]

        # Example: Get all relationships
        rels_result = session.run("CALL db.relationshipTypes()")
        relationships = [record["relationshipType"] for record in rels_result]

        schema = {
            "labels": labels,
            "relationships": relationships
        }

    return schema

# Convert schema to string format for prompt
def format_schema(schema):
    labels = "\n".join(f"  - {label}" for label in schema["labels"])
    relationships = "\n".join(f"  - {relationship}" for relationship in schema["relationships"])
    return f"Labels:\n{labels}\nRelationships:\n{relationships}"


# Connect to Neo4j database
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
schema = get_schema_from_neo4j(driver)



CYPHER_GENERATION_TEMPLATE = """
You are an assistant helping to generate Cypher queries for a Neo4j database. 
Given the user's question and the database schema provided below, generate a Cypher query that answers the question. 
Only output the Cypher query without any explanation or additional text.

data in knowledge graph :
**Nodes:**
- `Company`: Represents each company.
  - Properties: `name` (String), `symbol` (String)
- `FinancialStatement`: Represents the financial statement details of each company.
  - Properties: `year` (String), `quarter` (String)
- `MarketData`: Represents stock market-related data for the companies.
  - Properties: `date` (String)
- `Period`: Represents the time period for the financial data.
  - Properties: `year` (String), `quarter` (String), `date` (String)
- **Financial Nodes:**
  - `Assets`: Total assets.
    - Properties: `totalAssets` (Float)
  - `Liabilities`: Total liabilities.
    - Properties: `totalLiabilities` (Float)
  - `Equity`: Shareholder equity and total equity.
    - Properties: `shareholderEquity` (Float), `totalEquity` (Float)
  - `Revenue`: Revenue data.
    - Properties: `totalRevenueQuarter` (Float), `totalRevenueAccum` (Float)
  - `Expenses`: Expenses data.
    - Properties: `totalExpensesQuarter` (Float), `totalExpensesAccum` (Float)
  - `CashFlow`: Cash flow data.
    - Properties: `operatingCashFlow` (Float), `investingCashFlow` (Float), `financingCashFlow` (Float)
  - `FinancialRatio`: Various financial ratios.
    - Properties: `type` (String), `value` (Float)
    - Types: 
      - `ROE` (Return on Equity)
      - `ROA` (Return on Assets)
      - `DE` (Debt to Equity Ratio)
      - `PE` (Price to Earnings Ratio)
      - `PBV` (Price to Book Value)
  - `MarketRatio`: Various market ratios.
    - Properties: `type` (String), `value` (Float)
    - Types:
      - `PE` (Price to Earnings Ratio)
      - `PB` (Price to Book Ratio)
      - `Dividend_Yield` (Dividend_Yield)
      - `Beta` (Volatility Measure)
      - `Market_Cap`
      - `Volume_Turnover`
- **Market Data Nodes:**
  - `Price`: Price-related data.
    - Properties: `prior` (Float), `open` (Float), `high` (Float), `low` (Float), `close` (Float), `average` (Float)
  - `Volume`: Volume-related data.
    - Properties: `aomVolume` (Float), `aomValue` (Float), `trVolume` (Float), `trValue` (Float), `totalVolume` (Float), `totalValue` (Float)
  - `MarketRatio`: Various market ratios.
    - Properties: `type` (String), `value` (Float)
**Relationships:**
- `Company` to `FinancialStatement`: `HAS_FINANCIAL_STATEMENT`
- `FinancialStatement` to Financial Nodes: `HAS_ASSETS`, `HAS_LIABILITIES`, `HAS_EQUITY`, `HAS_REVENUE`, `HAS_EXPENSES`, `HAS_CASH_FLOW`, `HAS_RATIO`
- `FinancialStatement` to `Period`: `IS_FOR_PERIOD`
- `Company` to `MarketData`: `HAS_MARKET_DATA`
- `MarketData` to Market Data Nodes: `HAS_PRICE`, `HAS_VOLUME`, `HAS_RATIO`
- `MarketData` to `Period`: `IS_FOR_PERIOD`


Fine Tuning:
1.For stock tickers or company names, ensure that you follow the proper case sensitivity and return values as they appear in the database.
2.useing data in knowledge graph for write query
3.directly select value that want to know if database have, don't calculate it
4.when write query only use English
5.if you want to use price use like this (p:Price{{symbol: "ADVANC"}})
6.if you want to use MarketRatio use like this (m:MarketRatio{{symbol: "ADVANC"}})
7.Ensure the query is valid and aligned with the provided schema. If the query cannot be generated, return an explanation instead of leaving it blank.
  
Example Cypher Statements:
1.อัตราส่วนการคืนทุน (ROE) ของบริษัทADVANC ในไตรมาสที่ 1 ของปี 2019 คือเท่าไหร่:
```
MATCH (c:Company {{symbol: 'ADVANC'}})-[:HAS_FINANCIAL_STATEMENT]->(fs:FinancialStatement {{year: '2019', quarter: '1'}})
MATCH (fs)-[:HAS_RATIO]->(r:FinancialRatio {{type: 'ROE'}})
RETURN r.value AS ROE
```

2.สัดส่วนหนี้สินต่อทุน (DE) ของบริษัทADVANC ในไตรมาสที่ 1 ของปี 2019 คือเท่าไหร่:
```
MATCH (c:Company {{symbol: 'ADVANC'}})-[:HAS_FINANCIAL_STATEMENT]->(fs:FinancialStatement {{year: '2019', quarter: '1'}})
MATCH (fs)-[:HAS_RATIO]->(r:FinancialRatio {{type: 'DE'}})
RETURN r.value AS DebtToEquity
```

3.ราคาปิดของหุ้นADVANC ในวันที่ 1 กันยายน 2023 คือเท่าไหร่:
```
MATCH (md:MarketData{{date: "2023-09-01"}})-[:HAS_PRICE]->(p:Price{{symbol: "ADVANC"}})
RETURN p.close AS closingPrice
```

4.อัตราการจ่ายเงินปันผล (Dividend Yield) ของหุ้น BBL ในวันที่ 1 กันยายน 2023 คือเท่าไหร่:
```
MATCH (mr:MarketRatio{{symbol: 'BBL',type: "Dividend Yield",date: "2023-09-01"}})
RETURN mr.value as DividendYield
```

5.แนวโน้มของรายได้รวมของบริษัท ADVANC มีการเปลี่ยนแปลงอย่างไรในช่วงเวลาที่ผ่านมาจากข้อมูลทางการเงิน:
```
MATCH (c:Company {{symbol: 'ADVANC'}})-[:HAS_FINANCIAL_STATEMENT]->(fs:FinancialStatement)
MATCH (fs)-[:HAS_REVENUE]->(r:Revenue)
RETURN fs.year, fs.quarter, r.totalRevenueAccum
ORDER BY fs.year, fs.quarter
```

6.เปรียบเทียบผลตอบแทนต่อผู้ถือหุ้น (ROE) ของบริษัท ADVANC ในปี 2019 กับปี 2022:
```
MATCH (fs_2019:FinancialStatement {{year: '2019'}})-[:HAS_RATIO]->(roe_2019:FinancialRatio {{symbol: 'ADVANC',type: 'ROE'}})
MATCH (fs_2022:FinancialStatement {{year: '2022'}})-[:HAS_RATIO]->(roe_2022:FinancialRatio {{symbol: 'ADVANC',type: 'ROE'}})
RETURN roe_2019.value AS ROE_2019, roe_2022.value AS ROE_2022
```

7.เปรียบเทียบรายได้รวมในไตรมาสของบริษัท ADVANC กับบริษัท CPALL ในไตรมาสที่ 3 ปี 2021:
```
MATCH (adv_fs:FinancialStatement {{year: '2021', quarter: '3'}})-[:HAS_REVENUE]->(adv_r:Revenue{{symbol: 'ADVANC'}})
MATCH (cpall_fs:FinancialStatement {{year: '2021', quarter: '3'}})-[:HAS_REVENUE]->(cpall_r:Revenue{{symbol: 'CPALL'}})
RETURN adv_r.totalRevenueQuarter AS ADVANC_TotalRevenue, cpall_r.totalRevenueQuarter AS CPALL_TotalRevenue
```

Schema:
{schema}
Question:
{question}

Cypher query:
"""

# Create the prompt template for Cypher query generation
cypher_prompt = PromptTemplate(template=CYPHER_GENERATION_TEMPLATE,input_variables=["question"])
# Fetch schema from the Neo4j databas
schema_str = format_schema(schema)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph, 
    verbose=True,
    return_intermediate_steps=True,
    cypher_prompt=cypher_prompt
)



# Define a function to handle Cypher queries
def cypher_qa_function(input_text):
    try:
        # Fetch schema from the Neo4j databas
        schema_str = format_schema(schema)

        start_query_gen_time = time.time()
        generated_result = cypher_qa.invoke({"schema": schema_str, "query": input_text})
        end_query_gen_time = time.time()

        query = generated_result['intermediate_steps'][0]['query'].strip()

        if not query:
            return {
                "data": pd.DataFrame(),
                "query": None,
                "query_generation_time": end_query_gen_time - start_query_gen_time,
                "database_fetch_time": 0.0,
                "error": "Generated query is empty.",
            }

        start_db_time = time.time()
        with driver.session() as session:
            result = session.run(query)
            data = [record.data() for record in result]
        end_db_time = time.time()

        return {
            "data": data,
            "query": query,
            "query_generation_time": end_query_gen_time - start_query_gen_time,
            "database_fetch_time": end_db_time - start_db_time,
            "error": None,
        }

    except Exception as e:
        error_message = str(e)
        return {
            "data": pd.DataFrame(),
            "query": query if 'query' in locals() else None,
            "query_generation_time": end_query_gen_time - start_query_gen_time if 'end_query_gen_time' in locals() else 0.0,
            "database_fetch_time": end_db_time - start_db_time if 'end_db_time' in locals() else 0.0,
            "error": error_message,
        }


# Export the function
__all__ = ["cypher_qa_function"]

