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

Fine Tuning:
    1.data in knowledge graph :
        Node Types
        - `Company`: Node for company details, including attributes such as `symbol`, `name`.
        - `MarketData`: Node for daily company stock price data, including attributes such as `symbol`, `year`, `quarter`, `date`, prior,`open`, `high`, `low`, `close`, average, aomVolume, aomValue, trVolume, trValue, totalVolume, totalValue.
        - `Metric`: Node for company financial data by quarter, including attributes such as `symbol`, `year`, `quarter`, `date`, `type`, and `value`. type values: TotalAssets, TotalLiabilities, PaidupShareCapital, ShareholderEquity, TotalEquity, TotalRevenueQuarter, TotalRevenueAccum, TotalExpensesQuarter, TotalExpensesAccum, EBITQuarter, EBITAccum, NetProfitQuarter, NetProfitAccum,EPSQuarter,EPSAccum,OperatingCashFlow,InvestingCashFlow,FinancingCashFlow.
        - `Ratio`: Node for both financial and market ratios, including attributes such as `symbol`, `year`, `quarter`, `date`, `type`, and `value`. type values: ROE, ROA, NetProfitMarginQuarter, NetProfitMarginAccum, DE, FixedAssetTurnover, TotalAssetTurnover, PE, PBV, BVPS, DividendYield, MarketCap, VolumeTurnover.

        Relationships
        - `(:Company)-[:HAS_MARKET_DATA]->(:MarketData)` : Links a company to its stock market data.
        - `(:Company)-[:HAS_METRIC]->(:FinancialMetrics)` : Connects a company to its financial metrics.
        - `(:Company)-[:HAS_RATIO]->(:Ratio)` : Associates a company with its financial and market ratios.
        - `(:Company)-[:FREQUENTLY]->(:Ratio or :FinancialMetrics)`: Indicates key financial and market metrics frequently analyzed.
        - `(:Company {symbol: 'AOT'})` labeled as `PopularCompany`: Identifies leading companies with high market interest, including `PTT`, `BDMS`, `SCB`, and `CPALL`.

    2.Fine Tuning:
        - For stock tickers or company names, ensure that you follow the proper case sensitivity and return values as they appear in the database.
        - useing data in knowledge graph for write query
        - directly select value that want to know if database have, don't calculate it
        - when write query only use English
        - Ensure the query is valid and aligned with the provided schema. If the query cannot be generated, return an explanation instead of leaving it blank.
        - Do not add any text before or after the Cypher query. Only output the Cypher query.
        
    3.Example Cypher Statements:
        - Question: บริษัท AOT มีสินทรัพย์รวมในไตรมาสที่ 1 ปี 2019 เท่าไหร่
          Cypher Query:     ```
            MATCH (c:Company {symbol: 'AOT'})-[:HAS_METRIC]->(m:Metric {type: 'TotalAssets', year: '2019', quarter: '1'})
            RETURN m.value AS TotalAssets
            ```
        - Question: อัตรากำไรสุทธิ (Net Profit Margin) ของบริษัท PTT ในไตรมาสที่ 1 ปี 2019 คือเท่าไหร่
          Cypher Query:     ```
            MATCH (c:Company {symbol: 'PTT'})-[:HAS_RATIO]->(r:Ratio {type: 'NetProfitMarginQuarter', year: '2019', quarter: '1'})
            RETURN r.value AS NetProfitMargin
            ```

Schema:
{schema}
Question:
{question}

Cypher query:
"""


from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser

# Create the prompt template for Cypher query generation
cypher_prompt = PromptTemplate(template=CYPHER_GENERATION_TEMPLATE,input_variables=["question"])
# Fetch schema from the Neo4j databas
schema_str = format_schema(schema)

from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph, 
    verbose=True,
    return_intermediate_steps=True,
    cypher_prompt=cypher_prompt
)



# Define a function to handle Cypher queries
def financial_statements_function(input_text):
    try:
        # Fetch schema from the Neo4j databas
        schema_str = format_schema(schema)

        start_query_gen_time = time.time()
        generated_result = cypher_qa.invoke({"schema": schema_str, "query": input_text})
        end_query_gen_time = time.time()
        
        print(generated_result)
        
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
            "data": data,
            "query": query if 'query' in locals() else None,
            "query_generation_time": end_query_gen_time - start_query_gen_time if 'end_query_gen_time' in locals() else 0.0,
            "database_fetch_time": end_db_time - start_db_time if 'end_db_time' in locals() else 0.0,
            "error": error_message,
        }


# Export the function
__all__ = ["financial_statements_function"]

