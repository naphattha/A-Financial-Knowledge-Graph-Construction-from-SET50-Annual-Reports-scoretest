import streamlit as st

# Connect to Neo4j
import mysql.connector

host=st.secrets["MYSQL_HOST"]
user=st.secrets["MYSQL_USER"]
password=st.secrets["MYSQL_PASSWORD"]
database=st.secrets["MYSQL_DB"]

database = mysql.connector.connect(
        host=st.secrets["MYSQL_HOST"],
        user=st.secrets["MYSQL_USER"],
        password=st.secrets["MYSQL_PASSWORD"],
        database=st.secrets["MYSQL_DB"]
    )

# Construct the database URL
db_url = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
