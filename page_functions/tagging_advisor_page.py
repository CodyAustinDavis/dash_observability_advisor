import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import os
import dash_ag_grid as dag
from datetime import date, datetime, timedelta
from data_functions.backend_database import (
    QueryManager, 
    Base,
    AppComputeTags,
    TagPolicies,
    get_product_category_filter_tuples,
    get_start_ts_filter_min,
    get_end_ts_filter_max,
    get_tag_policy_name_filter,
    get_tag_policy_key_filter,
    get_tag_policy_value_filter,
    get_cluster_id_category_filter,
    get_job_id_category_filter,
    read_sql_file,
    execute_sql_from_file
)
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text



###### We want to load the data on app start up for filters so the UX is not slow when we switch tabs
"""
###### TO OD: 
1. Update Filters to only load the distinct values that are currently selected in the other filters
"""

# Load environment variables from .env file
# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
env_path = os.path.join(base_dir, '../', 'config', '.env')
# Load the environment variables
load_dotenv(dotenv_path=env_path)


## load auth into variables
host = os.getenv("DATABRICKS_SERVER_HOSTNAME")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
access_token = os.getenv("DATABRICKS_TOKEN")
catalog = os.getenv("DATABRICKS_CATALOG")
schema = os.getenv("DATABRICKS_SCHEMA")

##### Load Init Scripts for Data Tagging Advisor

## Run init scripts on app start-up

## Create Engine on start up
### System query manager uses the system catalog and schema scope (for the tables it creates and manages)
system_query_manager = QueryManager(host=host, http_path=http_path, access_token= access_token, catalog= catalog, schema = schema)
system_engine = system_query_manager.get_engine()

## Create all init tables that this app needs to exists - We use SQL Alchemy models so we can more easily programmatically write back
Base.metadata.create_all(system_engine, checkfirst=True)

#### Get Tables to Manage and Write To
TagPoliciesAGGRid = system_query_manager.reflect_table('app_tag_policies')


#### For the Reflection / AG Grid updates
def get_tag_policies_grid_data():
    session = system_query_manager.get_new_session()
    try:
        query = session.query(TagPoliciesAGGRid)
        df = pd.read_sql(query.statement, session.bind)
    finally:
        session.close()
    return df #.to_dict('records')


#### Run init SQL script to make sure all required tables are created
sql_init_filepath = './config/init.sql'  # Ensure the path is correct
execute_sql_from_file(system_engine, sql_init_filepath)


##### Load Parameter Filters
### Create a non-system query manager to read from tables in any catalog/schema
query_manager = QueryManager(host=host, http_path=http_path, access_token= access_token)
query_engine = query_manager.get_engine()


## load date filters
df_date_min_filter = pd.to_datetime(get_start_ts_filter_min(system_engine).iloc[0, 0]).date()
df_date_max_filter = pd.to_datetime(get_end_ts_filter_max(system_engine).iloc[0, 0]).date()

# Get today's date
today = datetime.now()
# Subtract 7 days
date_30_days_ago = today - timedelta(days=30)
current_date_filter = today.date()
day_30_rolling_filter = date_30_days_ago.date()

## load product category filters
df_product_cat_filter = get_product_category_filter_tuples(system_engine)
df_cluster_id_filter = get_cluster_id_category_filter(system_engine)
df_job_id_filter = get_job_id_category_filter(system_engine)
#df_tag_policy_filter = get_tag_policy_name_filter(system_engine)
#df_tag_key_filter = get_tag_policy_key_filter(system_engine)
#df_tag_value_filter = get_tag_policy_value_filter(system_engine)

# Fetch distinct tag policy names within the context manager

# Create a session factory with autocommit enabled
with QueryManager.session_scope(system_engine) as session:

    distinct_tag_policies = pd.read_sql(session.query(TagPolicies.tag_policy_name).distinct().statement, con=system_engine)
    distinct_tag_keys = pd.read_sql(session.query(TagPolicies.tag_key).distinct().statement, con=system_engine)
    distinct_tag_values = pd.read_sql(session.query(TagPolicies.tag_value).distinct().statement, con=system_engine)

    tag_policy_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_policies['tag_policy_name']]
    tag_key_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_keys['tag_key']]
    tag_value_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_values['tag_value']]


#### Tagging Advisor Page Function
def render_tagging_advisor_page():


    cellStyle = {
        "styleConditions": [
        {"condition": "highlightEdits(params)", "style": {"color": "orange"}},
    ]
    }

    defaultColDef = {
    "valueSetter": {"function": "addEdits(params)"},
    "editable": True,
    "cellStyle": cellStyle,
    }


    ##### REDER TAGGING PAGE
    layout = html.Div([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H1("Databricks Tagging Manager", style={'color': '#002147'}),  # A specific shade of blue
                ], width=8),
                dbc.Col([
                    html.Div([
                        html.Label('Tag Match Filter', style={'font-weight': 'bold', 'color': '#002147'}),
                        # RadioItems component for the filter
                            dcc.Dropdown(
                                id='tag-filter-dropdown',
                                options=[
                                    {'label': 'Matched', 'value': 'Matched'},
                                    {'label': 'Not Matched', 'value': 'Not Matched'},
                                    {'label': 'All', 'value': 'All'}
                                ],
                                value='All', 
                                multi=False,
                                clearable=False  # Prevents user from clearing the selection, ensuring a selection is always active
                            , style={'margin-bottom': '2px', 'margin-top': '10px'}),
                            # Output component to display the result based on the selected filter
                            html.Div(id='filter-output')
                    ])
                ], width=2),
                dbc.Col([
                    html.Button('Update Parameters', id='update-params-button', n_clicks=0, className = 'prettier-button'),  # A specific shade of blue
                ], width=2)
            ]),
            html.Div(className='border-top'),
            dbc.Row([
                    dbc.Col([
                    html.P("This app allows users to create tag policies, enforce those policies, and audit their Databricks usage to easily govern and categorize their usage in a way the fits their specific business needs.", style={'color': '#002147', 'margin-top': '10px'}),  # A specific shade of blue
                ], width=12)
            ])

        ], style={'margin-bottom': '10px'}),
        html.Div(className='border-top'),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([html.Label("Start Date", htmlFor='start-date-picker', style={'font-weight': 'bold'})], style={'margin-bottom': '5px', 'margin-top': '5px','margin-left': '5px', 'margin-right': '10px'})
                ], width=1),
                dbc.Col([
                    html.Div([html.Label("End Date", htmlFor='end-date-picker', style={'font-weight': 'bold'})], style={'margin-bottom': '5px', 'margin-top': '5px', 'margin-left': '10px'})
                ], width=2),
                dbc.Col([
                    html.Div([html.Div([html.Label("Product Category", htmlFor='product-category-dropdown', style={'font-weight': 'bold'})], style={'margin-bottom': '5px', 'margin-top': '5px'})
                    ], style={'margin-bottom': '5px', 'margin-top': '5px'})  # Adds spacing below each filter
                ], width=3),
                dbc.Col([
                    html.Div([html.Div([html.Label("Tag Policy Name", htmlFor='tag-policy-dropdown', style={'font-weight': 'bold'})], style={'margin-bottom': '5px', 'margin-top': '5px'})
                    ], style={'margin-bottom': '5px', 'margin-top': '5px'})  # Adds spacing below each filter
                ], width=2),
                dbc.Col([
                    html.Div([html.Div([html.Label("Tag Policy Key", htmlFor='tag-policy-key-dropdown', style={'font-weight': 'bold'})], style={'margin-bottom': '5px', 'margin-top': '5px'})
                    ], style={'margin-bottom': '5px', 'margin-top': '5px'})  # Adds spacing below each filter
                ], width=2),
                dbc.Col([
                    html.Div([html.Div([html.Label("Tag Policy Value", htmlFor='tag-policy-value-dropdown', style={'font-weight': 'bold'})], style={'margin-bottom': '5px', 'margin-top': '5px'})
                    ], style={'margin-bottom': '5px', 'margin-top': '5px'})  # Adds spacing below each filter
                ], width=2)

            ], style={'margin-bottom': '5px', 'margin-top': '5px'}),

            ### Filters
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.DatePickerSingle(
                            id='start-date-picker',
                            min_date_allowed= df_date_min_filter,
                            max_date_allowed= df_date_max_filter,
                            initial_visible_month=current_date_filter,
                            date=str(day_30_rolling_filter)
                        )
                    ], style={'margin-bottom': '2px', 'margin-top': '2px', 'margin-left': '5px', 'margin-right': '20px'})
                ], width=1),
                dbc.Col([
                    html.Div([
                        dcc.DatePickerSingle(
                            id='end-date-picker',
                            min_date_allowed= df_date_min_filter, 
                            max_date_allowed= df_date_max_filter,
                            initial_visible_month=current_date_filter,
                            date=str(current_date_filter)
                        )
                    ], style={'margin-bottom': '2px', 'margin-top': '2px', 'margin-left': '10px'})
                ], width=2),
                dbc.Col([
                    html.Div([
                        dcc.Dropdown(
                            id='product-category-dropdown',
                            options=[{'label': alias, 'value': category} for category, alias in zip(df_product_cat_filter['product_category'], df_product_cat_filter['product_category'])],
                            placeholder="Select a Product Category",
                            multi=True
                        )
                    ], style={'margin-bottom': '2px', 'margin-top': '2px'})  # Adds spacing below each filter
                ], width=3),
                dbc.Col([
                    html.Div([
                        dcc.Dropdown(
                            id='tag-policy-dropdown',
                            options=tag_policy_filter,
                            placeholder="Tag Policy Name",
                            multi=True
                        )
                    ], style={'margin-bottom': '2px', 'margin-top': '2px'})  # Adds spacing below each filter
                ], width=2),
                dbc.Col([
                    html.Div([
                        dcc.Dropdown(
                            id='tag-policy-key-dropdown',
                            options=tag_key_filter,
                            placeholder="Tag Key Name",
                            multi=True
                        )
                    ], style={'margin-bottom': '2px', 'margin-top': '2px'})  # Adds spacing below each filter
                ], width=2),
                dbc.Col([
                    html.Div([
                        dcc.Dropdown(
                            id='tag-policy-value-dropdown',
                            options=tag_value_filter,
                            placeholder="Tag Value Name",
                            multi=True
                        )
                    ], style={'margin-bottom': '2px', 'margin-top': '2px'})  # Adds spacing below each filter
                ], width=2),
            ], style={'margin-bottom': '2px', 'margin-top': '2px'})], id='filter-panel'),
        ## Build Visual Layer
        html.Div([

            ## Number Summary Card
            dbc.Row([
                dbc.Col([
                    html.Div([
                            dcc.Loading(
                                id="loading-matched-usage-chart",
                                type="default",  # Can be "graph", "cube", "circle", "dot", or "default"
                                children=html.Div([dcc.Graph(id='matched-usage-ind', className = 'chart-visuals')]),
                                fullscreen=False,  # True to make the spinner fullscreen
                                color= '#002147')
                            ])
                ], width=3),
                dbc.Col([
                    html.Div([
                            dcc.Loading(
                                id="loading-unmatched-usage-chart",
                                type="default",  # Can be "graph", "cube", "circle", "dot", or "default"
                                children=html.Div([dcc.Graph(id='unmatched-usage-ind', className = 'chart-visuals')]),
                                fullscreen=False,  # True to make the spinner fullscreen
                                color= '#002147')
                            ])
                ], width=3),
                            # Two new indicators or other contents to balance the layout
                dbc.Col([
                html.Div([
                    dcc.Loading(
                        id="loading-percent-match-ind",  # new ID for additional content
                        type="default",
                        children=html.Div([dcc.Graph(id='percent-match-ind', className='chart-visuals')]),
                        fullscreen=False,
                        color='#002147')
                    ])
                ], width=3),

                dbc.Col([
                html.Div([
                    dcc.Loading(
                        id="loading-total-usage-ind",  # new ID for additional content
                        type="default",
                        children=html.Div([dcc.Graph(id='total-usage-ind', className='chart-visuals')]),
                        fullscreen=False,
                        color='#002147')
                ])
                ], width=3),
                dbc.Col([
                    html.Div([
                            dcc.Loading(
                                id="loading-tag-chart",
                                type="default",  # Can be "graph", "cube", "circle", "dot", or "default"
                                children=html.Div([dcc.Graph(id='usage-by-match-chart', className = 'chart-visuals')]),
                                fullscreen=False,  # True to make the spinner fullscreen
                                color= '#002147')
                            ])
                        ], width=12),
            ], id='output-data'),

            ### Chart Row (2 Charts)
            dbc.Row([
                dbc.Col([
                    html.Div([
                            dcc.Loading(
                                id="loading-tag-histogram-total",
                                type="default",  # Can be "graph", "cube", "circle", "dot", or "default"
                                children=html.Div([dcc.Graph(id='usage-heatmap', className = 'chart-visuals')]),
                                fullscreen=False,  # True to make the spinner fullscreen
                                color= '#002147')
                            ])
                        ], width=6),
                dbc.Col([
                    html.Div([
                            dcc.Loading(
                                id="loading-tag-histogram",
                                type="default",  # Can be "graph", "cube", "circle", "dot", or "default"
                                children=html.Div([dcc.Graph(id='usage-by-tag-value-chart', className = 'chart-visuals')]),
                                fullscreen=False,  # True to make the spinner fullscreen
                                color= '#002147')
                            ])
                        ], width=6)
            ], id='output-data')
        ]
        ),
        html.Div(className='border-top'),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H3("Tag Audit Management", style={'color': '#002147'}),  # A specific shade of blue
                ], width=12)
            ]),
            html.Div(className='border-top'),
            dbc.Row([
                    dbc.Col([
                    html.P("This section allows users to create & manage tagging policies, find clusters/warehouses as well as jobs that are not properly tagged into the policies. \n Once these entities are found, they can then be properly categorized and tagging in the app.", style={'color': '#002147'}),  # A specific shade of blue
                ], width=12)
            ]),

            ### AG Grids!
            ## When "Save Policy Change is pushed, callback must update filter selections as well as the UI for the Grid
            ## The callback must also show a ! logo when there are unsaved changes
            ## The logo then needs to clear the Store when changed are saved and reloaded
            dbc.Row([
                    dbc.Col([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                    dcc.Loading(
                                    id="loading-save-policies",
                                    type="default",  # Choose the style of the loading animation
                                    color= '#002147', 
                                    children=html.Button('Save Policy Changes', id='tag-policy-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                                )
                            ], width=4),
                            dbc.Col([
                                dcc.Loading(
                                    id="loading-clear-policies",
                                    type="default",  # Choose the style of the loading animation
                                    color= '#002147',
                                    children=html.Button('Clear Policy Changes', id='tag-policy-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                                )
                            ], width=4),
                            dbc.Col([
                                dcc.Loading(
                                    id="loading-add-policies",
                                    type="default",  # Choose the style of the loading animation
                                    color= '#002147',
                                    children=html.Button('Add +', id='add-policy-row-btn', n_clicks=0, style={'margin-bottom': '10px', 'font-size': '14px'},
                                    className = 'add-button')
                                )
                            ], width=2),
                             dbc.Col([
                                dcc.Loading(
                                    id="loading-remove-policies",
                                    type="default",  # Choose the style of the loading animation
                                    color= '#002147',
                                    children=html.Button('Remove -', id='remove-policy-row-btn', n_clicks=0, style={'margin-bottom': '10px', 'font-size': '14px'}, className = 'remove-button')
                                )
                            ], width=2)
                        ]),
                        html.Div(id='policy-change-indicator', children=[
                                html.Span("! Pending Changes to be Saved", style={'color': '#DAA520'}),  # Dark yellow color
                            ], style={'display': 'none'}),
                        ## DEGBUG -- html.Div(id='changes-display'),  # This Div will show the changes
                            dag.AgGrid(
                                id='tag-policy-ag-grid',


                                columnDefs=[
                                    {
                                        'headerName': 'Policy Id', 
                                        'field': 'tag_policy_id', 
                                        'editable': False, 
                                        'width': 100, 
                                        'suppressSizeToFit': True
                                    },
                                    {
                                        'headerName': 'Policy Name', 
                                        'field': 'tag_policy_name', 
                                        'editable': True,
                                        'suppressSizeToFit': True
                                    },
                                    {
                                        'headerName': 'Tag Key', 
                                        'field': 'tag_key', 
                                        'width': 150, 
                                        'editable': True,
                                        'suppressSizeToFit': True
                                    },
                                    {
                                        'headerName': 'Tag Value', 
                                        'field': 'tag_value', 
                                        'editable': True,
                                        'enableCellChangeFlash': True,
                                        'suppressSizeToFit': True,
                                        'cellStyle': {
                                            "styleConditions": [
                                                {"condition": "dagfuncs.highlightEdits(params)", "style": {"backgroundColor": "yellow"}}
                                            ]
                                        }
                                    },
                                    {'headerCheckboxSelection': True, 'checkboxSelection': True, 'headerCheckboxSelectionFilteredOnly': True, 'width': 50, 
                                    'suppressSizeToFit': True},
                                ],
                                defaultColDef=defaultColDef,
                                rowData=get_tag_policies_grid_data().to_dict('records'),
                                dashGridOptions={
                                    'enableRangeSelection': True,
                                    'rowSelection': 'multiple'
                                    }
                            ),
                            dcc.Store(id='policy-changes-store')
                        ])
                ], width=5, style={'margin-left':'10px', 'margin-right':'10px'})
            ])
        ]),

    html.Div(id='output-container')
    ])

    return layout



