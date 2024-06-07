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
    func,
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
        session = sessionmaker(bind=engine, expire_on_commit=True)()
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





##### Load Dynamic SQL Files for TAG_QUERY
def build_tag_query_from_params(
                                start_date, end_date, 
                                tag_filter = None,
                                product_category = None, 
                                tag_policies = None, tag_keys = None, tag_values = None, 
                                final_agg_query=None):
    
    ## Load Base Query Template Parts
    TAG_QUERY_1 = read_sql_file("./config/tagging_advisor/base_tag_query_1.sql") ## Where you add selected tags
    TAG_QUERY_2 = read_sql_file("./config/tagging_advisor/base_tag_query_2.sql")
    TAG_QUERY_3 = read_sql_file("./config/tagging_advisor/base_tag_query_3.sql") ## Where you filter the rest of the paramters
    TAG_QUERY_4 = read_sql_file("./config/tagging_advisor/base_tag_query_4.sql") ## Where you select the final data frame


    FINAL_QUERY = ""

    ## Tag Dynamic Filter Construction
    if tag_policies:

        tag_policies_str = ', '.join([f"'{key}'" for key in tag_policies]) 
        TAG_QUERY_1 = TAG_QUERY_1 + f"\n AND tag_policy_name IN ({tag_policies_str})"
    
    if tag_keys:
        tag_keys_str = ', '.join([f"'{key}'" for key in tag_keys]) 
        TAG_QUERY_1 = TAG_QUERY_1 + f"\n AND tag_key IN ({tag_keys_str})"
    
    if tag_values:
        tag_values_str = ', '.join([f"'{key}'" for key in tag_values]) 
        TAG_QUERY_1 = TAG_QUERY_1 + f"\n AND tag_value IN ({tag_values_str})"

    FINAL_QUERY = FINAL_QUERY + "\n" + TAG_QUERY_1 + "\n" + TAG_QUERY_2
    ## 
    FINAL_QUERY = FINAL_QUERY + TAG_QUERY_3 + f"\n AND usage_start_time >= '{start_date}'::timestamp \n AND usage_start_time <= '{end_date}'::timestamp"


    if product_category:
        product_categories_str = ', '.join([f"'{key}'" for key in product_category]) 
        FINAL_QUERY = FINAL_QUERY + f"\n AND billing_origin_product IN ({product_categories_str})"

    if tag_filter == 'Matched':
        FINAL_QUERY = FINAL_QUERY + f"\n AND IsTaggingMatch = 'In Policy' "
    elif tag_filter == 'Not Matched':
        FINAL_QUERY = FINAL_QUERY + f"\n AND IsTaggingMatch = 'Not Matched To Tag Policy' "
    else: 
        pass
    
    ## Final Select Statement -- this is after all the standard server-side filtering

    if final_agg_query:
        FINAL_QUERY = FINAL_QUERY + ")\n" + final_agg_query
    
    else:
        FINAL_QUERY = FINAL_QUERY + TAG_QUERY_4

    return FINAL_QUERY



# Adhoc AG Grid
def build_adhoc_ag_grid_from_params(
                                start_date, end_date, tag_filter = None,
                                tag_policies = None, tag_keys = None, tag_values = None, 
                                final_agg_query=None, top_n = None):
    
    ## Load Base Query Template Parts
    BASE_AGG_QUERY = """
                    SELECT 
                    clean_cluster_id AS cluster_id,
                    MAX(cluster_name) AS cluster_name,
                    MIN(IsTaggingMatch) AS is_tag_policy_match,
                    array_distinct(collect_list(MatchedTagValues)) AS tag_matches,
                    array_distinct(collect_list(MissingTagKeys)) AS missing_tags,
                    MAX(billing_origin_product) AS product_type,
                    MAX(workspace_id) AS workspace_id,
                    MAX(account_id) AS account_id,
                    last_value(clean_tags) AS tags,
                    MAX(clean_usage_owner) AS resource_owner,
                    round(SUM(usage_quantity), 2) AS usage_quantity,
                    round(SUM(Dollar_DBUs_List), 2) AS Dollar_DBUs_List,
                    round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 7  THEN Dollar_DBUs_List END), 2) AS T7_Usage,
                    round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 30  THEN Dollar_DBUs_List END), 2) AS T30_Usage,
                    round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 90  THEN Dollar_DBUs_List END), 2) AS T90_Usage,
                    MIN(usage_date) AS first_usage_date,
                    MAX(usage_date) AS latest_usage_date,
                    date_diff(DAY, first_usage_date , getdate()) AS resource_age,
                    date_diff(DAY, latest_usage_date , getdate()) AS days_since_last_use
                    FROM filtered_result
                    WHERE 1=1 
                    AND billing_origin_product IN ('ALL_PURPOSE')
                    GROUP BY clean_cluster_id
                    ORDER BY Dollar_DBUs_List DESC
                    """

    if top_n:
        top_n = int(top_n)
        BASE_AGG_QUERY = BASE_AGG_QUERY + f"\n LIMIT {top_n}"

    FINAL_QUERY = build_tag_query_from_params(start_date=start_date, end_date=end_date, tag_filter=tag_filter, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=BASE_AGG_QUERY)
    
    return FINAL_QUERY


#### SQL AG Grid
def build_sql_ag_grid_from_params(
                                start_date, end_date, tag_filter = None,
                                tag_policies = None, tag_keys = None, tag_values = None, 
                                final_agg_query=None, top_n = None):
    
    ## Load Base Query Template Parts
    BASE_AGG_QUERY = """
                    SELECT 
                    clean_warehouse_id AS warehouse_id,
                    MIN(IsTaggingMatch) AS is_tag_policy_match,
                    array_distinct(collect_list(MatchedTagValues)) AS tag_matches,
                    array_distinct(collect_list(MissingTagKeys)) AS missing_tags,
                    MAX(billing_origin_product) AS product_type,
                    MAX(workspace_id) AS workspace_id,
                    MAX(account_id) AS account_id,
                    last_value(clean_tags) AS tags,
                    MAX(clean_usage_owner) AS resource_owner,
                    round(SUM(usage_quantity), 2) AS usage_quantity,
                    round(SUM(Dollar_DBUs_List), 2) AS Dollar_DBUs_List,
                    round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 7  THEN Dollar_DBUs_List END), 2) AS T7_Usage,
                    round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 30  THEN Dollar_DBUs_List END), 2) AS T30_Usage,
                    round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 90  THEN Dollar_DBUs_List END), 2) AS T90_Usage,
                    MIN(usage_date) AS first_usage_date,
                    MAX(usage_date) AS latest_usage_date,
                    date_diff(DAY, first_usage_date , getdate()) AS resource_age,
                    date_diff(DAY, latest_usage_date , getdate()) AS days_since_last_use
                    FROM filtered_result
                    WHERE 1=1 
                    AND billing_origin_product IN ('SQL')
                    AND clean_warehouse_id IS NOT NULL
                    GROUP BY clean_warehouse_id
                    ORDER BY Dollar_DBUs_List DESC
                    """

    if top_n:
        top_n = int(top_n)
        BASE_AGG_QUERY = BASE_AGG_QUERY + f"\n LIMIT {top_n}"


    FINAL_QUERY = build_tag_query_from_params(start_date=start_date, end_date=end_date, tag_filter=tag_filter, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=BASE_AGG_QUERY)
    
    return FINAL_QUERY


#### JOBS AG Grid
def build_jobs_ag_grid_from_params(
                                start_date, end_date, tag_filter = None,
                                tag_policies = None, tag_keys = None, tag_values = None, 
                                final_agg_query=None, top_n = None):
    
    ## Load Base Query Template Parts
    BASE_AGG_QUERY = """
                    SELECT 
                    clean_job_or_pipeline_id AS job_id,
                    MAX(cluster_name) AS cluster_name,
                    MIN(IsTaggingMatch) AS is_tag_policy_match,
                    array_distinct(collect_list(MatchedTagValues)) AS tag_matches,
                    array_distinct(collect_list(MissingTagKeys)) AS missing_tags,
                    MAX(billing_origin_product) AS product_type,
                    MAX(workspace_id) AS workspace_id,
                    MAX(account_id) AS account_id,
                    last_value(clean_tags) AS tags,
                    MAX(clean_usage_owner) AS resource_owner,
                    round(SUM(usage_quantity), 2) AS usage_quantity,
                    round(SUM(Dollar_DBUs_List), 2) AS Dollar_DBUs_List,
                    round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 7  THEN Dollar_DBUs_List END), 2) AS T7_Usage,
                    round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 30  THEN Dollar_DBUs_List END), 2) AS T30_Usage,
                    round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 90  THEN Dollar_DBUs_List END), 2) AS T90_Usage,
                    MIN(usage_date) AS first_usage_date,
                    MAX(usage_date) AS latest_usage_date,
                    date_diff(DAY, first_usage_date , getdate()) AS resource_age,
                    date_diff(DAY, latest_usage_date , getdate()) AS days_since_last_use
                    FROM filtered_result
                    WHERE 1=1 
                    AND billing_origin_product IN ('JOBS')
                    AND clean_job_or_pipeline_id IS NOT NULL
                    GROUP BY clean_job_or_pipeline_id
                    ORDER BY Dollar_DBUs_List DESC
                    """

    if top_n:
        top_n = int(top_n)
        BASE_AGG_QUERY = BASE_AGG_QUERY + f"\n LIMIT {top_n}"

    FINAL_QUERY = build_tag_query_from_params(start_date=start_date, end_date=end_date, tag_filter=tag_filter, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=BASE_AGG_QUERY)
    return FINAL_QUERY

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
    tag_policy_name = Column(String, nullable = False)
    tag_key = Column(String, nullable = False)
    tag_value = Column(String)
    tag_policy_name = Column(String)
    update_timestamp = Column(TIMESTAMP)
    is_persisted_to_actual_asset = Column(Boolean)


class TagPolicies(Base):
    __tablename__ = "app_tag_policies"

    tag_policy_id = Column(BigInteger, Identity(always=True), primary_key = True, autoincrement=True)
    tag_policy_name = Column(String, nullable=False)
    tag_policy_description = Column(String, nullable=True)
    tag_key = Column(String, nullable=False)
    tag_value = Column(String, nullable=True)
    update_timestamp = Column(TIMESTAMP, nullable=False, default=func.now())
