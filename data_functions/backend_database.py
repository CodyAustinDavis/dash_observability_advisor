import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from databricks import sql
import os
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from uuid import UUID
from dotenv import load_dotenv
from databricks.sqlalchemy import TIMESTAMP, TINYINT
import pandas as pd
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.automap import automap_base
from typing import Dict
from contextlib import contextmanager

# Beside the CamelCase types shown below, line comments reflect
# the underlying Databricks SQL / Delta table type
from sqlalchemy import (
    BigInteger,      # BIGINT
    Boolean,         # BOOLEAN
    Column,
    Date,            # DATE
    DateTime,        # TIMESTAMP_NTZ
    Integer,         # INTEGER
    Numeric,         # DECIMAL
    String,          # STRING
    Time,            # STRING
    Uuid,            # STRING
    create_engine,
    select,
    text,
    MetaData,
    Identity
)
from sqlalchemy.orm import DeclarativeBase, Session


##### Query Management for executing arbitrary queries to the database with parameter bindings and post result filters on client side

class QueryManager:

    def __init__(self, host: str, http_path: str, access_token: str, catalog: str = None, schema: str = None):

        extra_connect_args = {
        "_tls_verify_hostname": True,
        "_user_agent_entry": "Databricks Observability Advisor",
        }

        ## url with catalog and schema optional
        database_url = f"databricks://token:{access_token}@{host}?http_path={http_path}"

        if catalog is not None:
            database_url = database_url + f"&catalog={catalog}"
        if schema is not None: 
            database_url = database_url + f"&schema={schema}"

        self.engine = create_engine(
            database_url,
            connect_args=extra_connect_args, echo=True,
        )

        self.connection_url = database_url
        self.session_factory = sessionmaker(bind=self.engine)

        # Create a scoped session
        self.db_session = scoped_session(self.session_factory)

    def get_engine(self):
        return self.engine
    
    def get_url(self):
        return self.connection_url

    def get_new_session(self):
        return self.db_session()
    

    def reflect_table(self, table_name):

        metadata = MetaData()
        metadata.reflect(self.engine, only=[table_name])
        Base = automap_base(metadata=metadata)
        Base.prepare()
        return Base.classes.get(table_name)
    

    def execute_query_to_df(self, sql, **bind_params):
        """
        Execute a complex SQL query with parameter support and return results in a pandas DataFrame.
        """
        session = self.get_new_session()  # Get a new session instance
        try:
            # Execute query with provided bind parameters
            df = pd.read_sql_query(sql, con=self.engine, params=bind_params)
            session.commit()  # Ensure any transactional changes are committed
        except Exception as e:
            session.rollback()  # Rollback in case of an exception
            print(f"An error occurred: {e}")
            raise
        finally:
            session.close()  # Ensure session is closed regardless of success or errors
        return df


    def apply_dataframe_filters(self, df, **filter_params):
        """
        Apply filters to a DataFrame based on provided parameters.
        """
        for key, value in filter_params.items():
            if value is not None:
                df = df[df[key] == value]
        return df
    
    def get_query_dataframe_results(self, sql, bind_params: Dict[str, str] = None, filter_params: Dict[str, str] = None):
        
        try: 
            df_param_result = self.execute_query_to_df(sql,  bind_params=bind_params)
            df_filtered = self.apply_dataframe_filters(df=df_param_result, filter_params=filter_params)

            return df_filtered
        
        except Exception as e:
            raise(str(e))
        
    @staticmethod
    @contextmanager
    def session_scope(engine):
        """Provide a transactional scope around a series of operations."""
        session = sessionmaker(bind=engine)()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
       


##### Define Functions for getting top level filters -- this will not be dynamic because I am lazy :)
#### TO DO: Make each tab all use 1 query to get this instead of a bunch
def get_cluster_id_category_filter(engine):
    query = "SELECT DISTINCT clean_cluster_id AS cluster_id FROM clean_usage WHERE clean_cluster_id IS NOT NULL"
    attributes = pd.read_sql_query(query, con=engine)
    return attributes


def get_job_id_category_filter(engine):
    query = "SELECT DISTINCT clean_job_or_pipeline_id AS job_id FROM clean_usage WHERE clean_job_or_pipeline_id IS NOT NULL"
    attributes = pd.read_sql_query(query, con=engine)
    return attributes

def get_product_category_filter_tuples(engine):
    query = "SELECT DISTINCT billing_origin_product AS product_category FROM clean_usage"
    attributes = pd.read_sql_query(query, con=engine)
    return attributes


def get_start_ts_filter_min(engine):
    query = "SELECT MIN(usage_date)::date AS usage_date FROM clean_usage"
    attributes = pd.read_sql_query(query, con=engine)
    return attributes

def get_end_ts_filter_max(engine):
    query = "SELECT MAX(usage_date)::date AS usage_date FROM clean_usage"
    attributes = pd.read_sql_query(query, con=engine)
    return attributes

def get_tag_policy_name_filter(engine):
    query = "SELECT DISTINCT tag_policy_name AS tag_policy_name FROM tag_policies"
    attributes = pd.read_sql_query(query, con=engine)
    return attributes

def get_tag_policy_key_filter(engine):
    query = "SELECT DISTINCT tag_key AS tag_key FROM tag_policies"
    attributes = pd.read_sql_query(query, con=engine)
    return attributes

def get_tag_policy_value_filter(engine):
    query = "SELECT DISTINCT tag_value AS tag_value FROM tag_policies"
    attributes = pd.read_sql_query(query, con=engine)
    return attributes


##### Utility fucntions
def read_sql_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def execute_sql_from_file(engine, filepath):
    # Read SQL from the file
    sql_commands = read_sql_file(filepath)
    # Split the SQL commands by ';' assuming that's your statement delimiter
    commands = sql_commands.split(';')
    
    with engine.connect() as connection:
        for command in filter(None, map(str.strip, commands)):
            command = str(command.strip())
             # Prepare the command text for execution
            sql_command = text(command)

            print(sql_command)
            if command:  # Make sure it's not an empty command
                connection.execute(sql_command)



##### Initial Data Model of App - this tables must exist for app to run so they are deployed and checked on start-up

## Define DDL Tables to implement on start-up

class Base(DeclarativeBase):
    pass


## This is the table that allows users to tag assets (cluters/jobs/pipelines/warehouses) in the app without affecting actual tags
# Users can sync with actual tags later via jobs API call that checks if they are synced
class AppComputeTags(Base):
    __tablename__ = "app_compute_tags"

    tag_id = Column(BigInteger, Identity(always=True), primary_key=True)
    compute_asset_id = Column(String, nullable = False)
    compute_asset_type = Column(String, nullable = False)
    tag_key = Column(String, nullable = False)
    tag_value = Column(String)
    tag_policy_name = Column(String)
    update_timestamp = Column(TIMESTAMP)
    is_persisted_to_actual_asset = Column(Boolean)


class TagPolicies(Base):
    __tablename__ = "app_tag_policies"

    tag_policy_id = Column(BigInteger, Identity(always=True), primary_key = True)
    tag_policy_name = Column(String, nullable=False)
    tag_policy_description = Column(String, nullable=True)
    tag_key = Column(String, nullable=False)
    tag_value = Column(String, nullable=True)
    update_timestamp = Column(TIMESTAMP, nullable=False)
