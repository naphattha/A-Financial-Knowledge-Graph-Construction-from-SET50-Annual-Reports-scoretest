# A Financial Knowledge Graph Construction from SET50 Annual Reports

This project constructs a Financial Knowledge Graph (KG) from the annual reports of SET50 companies. It enables users to query financial data through a chatbot interface, facilitating complex financial analysis. The project compares the performance and efficiency of two database systems: Neo4j (graph-based) and MySQL (relational).

---

## Project Structure

- `llm-chatbot-python_mysql - ENG`: Chatbot using MySQL with English question support system  
- `llm-chatbot-python_mysql - THAI`: Chatbot using MySQL with Thai question support system  
- `llm-chatbot-python_neo4j - ENG`: Chatbot using Neo4j with English question support system  
- `llm-chatbot-python_neo4j - THAI`: Chatbot using Neo4j with Thai question support system  

Each directory contains code and assets to run an independent chatbot instance in the specified language and database type.

---

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python**: Version 3.8 or higher  
- **Neo4j**: Community or Enterprise Edition  
- **MySQL**: Version 8.0 or higher  
- **Node.js and npm**: For frontend development (if applicable)

---

## Set Up Databases

To populate the databases with data, please follow the instructions in the database setup repository:  
👉 [A-Financial-Knowledge-Graph-SET50-Annual-Reports-database](https://github.com/naphattha/A-Financial-Knowledge-Graph-SET50-Annual-Reports-database)

---

## Installation

Clone the Repository:

```bash
git clone https://github.com/naphattha/A-Financial-Knowledge-Graph-Construction-from-SET50-Annual-Reports-scoretest.git
cd A-Financial-Knowledge-Graph-Construction-from-SET50-Annual-Reports-scoretest
```

Set Up a Virtual Environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application
Use the following files to run and evaluate the application:

test.ipynb (in each directory): Test chatbot responses using prepared question sets.

mysql_test.ipynb: Run direct queries on MySQL and record execution time.

neo4j_test.ipynb: Run direct queries on Neo4j and record execution time.

score_beleu&bert.ipynb: Evaluate chatbot answers using BLEU and BERTScore metrics with reference answers in Excel.
