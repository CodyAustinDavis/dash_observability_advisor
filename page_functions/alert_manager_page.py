import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import dash_ag_grid as dag
from datetime import date, datetime, time, timedelta, timezone
from chart_functions import ChartFormats
from visual_functions import (
    get_adhoc_ag_grid_column_defs
)
from agent_functions import *
from data_functions.backend_database import (
    QueryManager, 
    get_product_category_filter_tuples,
    get_start_ts_filter_min,
    get_end_ts_filter_max,
    get_tag_policy_name_filter,
    get_tag_policy_key_filter,
    get_tag_policy_value_filter
)
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
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql



#### Data Functions

# Function to fetch data from the database

class AlertsManager:

    def __init__(self, system_query_manager):
        self.system_query_manager = system_query_manager


    def get_alerts_ag_grid_data(self):
        engine = self.system_query_manager.get_engine()

        query = text("""
        SELECT id, alert_name, alert_query, alert_schedule, alert_recipients, query_id, alert_id, job_id, alert_column, alert_condition, alert_condition_value
        FROM alerts_settings
        """)
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        return df #.to_dict('records')



###### Render Full Page #####

def render_alert_manager_page(alerts_ag_grid_data):

    return dbc.Container([dbc.Row([
            dbc.Col(html.H1("Alert Manager", style={'color': '#002147'}), width=12),
        ]),
        html.Div(className='border-top'),
        dbc.Row([
            dbc.Col(dbc.Button('Submit', id='chat-submit-btn', n_clicks=0, class_name='prettier-button'), width=3, style = {'margin-bottom': '10px'}),
            dbc.Col(dbc.Button('Clear Context', id='clear-context-btn', n_clicks=0, class_name='prettier-button'), width=3, style = {'margin-bottom': '10px'}),
            dbc.Col(dcc.Input(id='chat-input-box', type='text', placeholder='What alerts would you like to set up?', className='form-control'), width=12)
        ]),
        dcc.Store(id='chat-history', data={'messages': []}),
        dcc.Store(id='in-progress-alert', data={'alert': []}),
        html.Div(className='border-top'),

        #### Ag Grid Title and Buttons (delete, save)
        dbc.Row([dbc.Col(html.H4("Saved Alerts", style={'color':'#002147', 'margin-top':'10px'}), width=2),
                 dbc.Col(width=6), ## spacer
                dbc.Col(dcc.Loading(
                                    id="loading-create-jobs-alerts",
                                    type="default",  # Choose the style of the loading animation
                                    children=html.Button('Create Alert Jobs', id='create-alert-jobs-btn', n_clicks=0,
                                    className = 'prettier-button', style={'margin-bottom': '10px'})
                                , style={'color':'#002147'}), width=2),
                 dbc.Col(dcc.Loading(
                                    id="loading-remove-alerts",
                                    type="default",  # Choose the style of the loading animation
                                    children=html.Button('Delete Rows', id='remove-alerts-btn', n_clicks=0,
                                    className = 'prettier-button', style={'margin-bottom': '10px'})
                                , style={'color':'#002147'}), width=2)
                 ]),
        dbc.Row([
            dbc.Col([
                    dag.AgGrid(
                        id='alerts-grid',
                        className='ag-theme-alpine',
                        columnDefs=[
                            {'headerCheckboxSelection': True, 'checkboxSelection': True, 'headerCheckboxSelectionFilteredOnly': True, 'width': 50, 'suppressSizeToFit': True},
                            {'headerName': "ID", 'field': "id", 'width': 80},
                            {'headerName': "Alert Name", 'field': "alert_name"},
                            {'headerName': "Alert Query", 'field': "alert_query"},
                            {'headerName': "Schedule", 'field': "alert_schedule"},
                            {'headerName': "Recipients", 'field': "alert_recipients"},
                            {'headerName': "Query ID", 'field': "query_id"},
                            {'headerName': "Alert ID", 'field': "alert_id"},
                            {'headerName': "Job ID", 'field': "job_id"},
                            {'headerName': "Alert Column", 'field': "alert_column"},
                            {'headerName': "Alert Condition", 'field': "alert_condition"},
                            {'headerName': "Alert Condition Value", 'field': "alert_condition_value"},
                        ],
                        rowData= alerts_ag_grid_data.to_dict('records'),  # This will be populated by a callback
                        defaultColDef={'sortable': True, 'filter': True, 'editable': False},
                    )
            ], width=12)
        ]),
        html.Div(className='border-top'),
        dbc.Row([dbc.Col(html.H4("Chat Window", style={'color':'#002147', 'margin-top':'10px'}), width=2),
                 dbc.Col(width=6), ## spacer
                 dbc.Col(html.H4("Pending Alert Data", style={'color':'#002147', 'margin-top':'10px'}), width=2),
                 dbc.Col(dcc.Loading(
                                    id="save-pending-alert-loading",
                                    type="default",  # Choose the style of the loading animation
                                    children=html.Button('Save Alert', id='save-pending-alerts-btn', n_clicks=0,
                                    className = 'prettier-button'), style={'color':'#002147'}
                                ), width=2)
                 ]),
        dbc.Row([
            dbc.Col(
                [
                    dcc.Loading(
                            id='loading-markdown',  # Unique identifier for the loading component
                            children=[dcc.Markdown(id='chat-output-window',
                                children='Please submit a chat to get started...',
                                style={
                                'white-space': 'pre-wrap', 
                                'color': 'white',
                                'background-color': '#002147', 
                                'border': '1px solid #002147', 
                                'margin-top': '10px',
                                'padding': '10px'
                            }, dangerously_allow_html=True)],
                            type='default',  # or you can use other types like 'circle', 'dot', or 'cube'
                            fullscreen=False,  # True to cover the whole viewport, False to cover only the children
                            color='#002147'  # Optional: color of the spinner
                        )
                ], width=8),
            dbc.Col(
                [
                        # Add the input components for the extracted values
                html.Div([
                    html.Label('Proposed Name:', style={'color': '#002147'}),
                    dcc.Input(id='input-alert-name', type='text', style={'width': '100%'}),
                    html.Label('Query:', style={'color': '#002147'}),
                    dcc.Input(id='input-query', type='text', style={'width': '100%'}),
                    html.Label('Schedule:', style={'color': '#002147'}),
                    dcc.Input(id='input-schedule', type='text', style={'width': '100%'}),
                    html.Label('Recipients:', style={'color': '#002147'}),
                    dcc.Input(id='input-recipients', type='text', style={'width': '100%'}),
                    html.Label('Context SQL:', style={'color': '#002147'}),
                    dcc.Input(id='input-context-sql', type='text', style={'width': '100%'})
                ])
                ]
            , width=4)
        ])
    ]
    ,fluid=True, style={'width': '100vw'})


