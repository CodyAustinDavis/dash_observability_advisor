import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import os
import dash_ag_grid as dag
from datetime import date, datetime, time, timedelta, timezone
from chart_functions import ChartFormats
from visual_functions import (
    get_adhoc_ag_grid_column_defs
)
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
    execute_sql_from_file,
    build_tag_query_from_params,
    build_adhoc_ag_grid_from_params,
    build_jobs_ag_grid_from_params,
    build_sql_ag_grid_from_params
)
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text



###### We want to load the data on app start up for filters so the UX is not slow when we switch tabs
"""
###### TO DO: 
1. Update Filters to only load the distinct values that are currently selected in the other filters
"""

#### Run init SQL script to make sure all required tables are created


class TagAdvisorPageManager():

    def __init__(self, system_query_manager):
        self.system_query_manager = system_query_manager

        return
    
    def run_init_scripts(self):
        system_engine = self.system_query_manager.get_engine()
        sql_init_filepath = './config/init.sql'  # Ensure the path is correct
        execute_sql_from_file(system_engine, sql_init_filepath)
        ## Create all init tables that this app needs to exists - We use SQL Alchemy models so we can more easily programmatically write back
        Base.metadata.create_all(system_engine, checkfirst=True)
        return


    #### For the Reflection / AG Grid updates
    def get_tag_policies_grid_data(self):

        TagPoliciesAGGRid = self.system_query_manager.reflect_table('app_tag_policies')
        session = self.system_query_manager.get_new_session()
        try:
            query = session.query(TagPoliciesAGGRid)
            df = pd.read_sql(query.statement, session.bind)

        finally:
            session.close()
        return df #.to_dict('records')


    #### Get Adhoc Usage AG Grid Data
    def get_adhoc_clusters_grid_data(self, start_date, end_date, tag_filter=None, tag_policies=None, product_category=None, tag_keys=None, tag_values=None, compute_tag_keys=None, top_n=100):

        query = build_adhoc_ag_grid_from_params(start_date=start_date, end_date=end_date, tag_filter= tag_filter, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, compute_tag_keys=compute_tag_keys, top_n=top_n)
        session = self.system_query_manager.get_new_session()
        try:

            df = pd.read_sql(text(query), session.bind)
            ## Need to ad a row index to track changes since this result does not have a clear primary key
            df['rowIndex'] = df.index

        finally:
            session.close()
        return df #.to_dict('records')


    #### Get Jobs Usage AG Grid Data
    def get_jobs_clusters_grid_data(self, start_date, end_date, tag_filter=None, product_category=None, tag_policies=None, tag_keys=None, tag_values=None, compute_tag_keys=None, top_n=100):

        query = build_jobs_ag_grid_from_params(start_date=start_date, end_date=end_date, tag_filter=tag_filter, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, compute_tag_keys=compute_tag_keys, top_n=top_n)
        session = self.system_query_manager.get_new_session()
        try:

            df = pd.read_sql(text(query), session.bind)
            df['rowIndex'] = df.index
        finally:
            session.close()
        return df #.to_dict('records')


    #### Get SQL Usage AG Grid Data
    def get_sql_clusters_grid_data(self, start_date, end_date, tag_filter=None, product_category = None, tag_policies=None, tag_keys=None, tag_values=None, compute_tag_keys=None, top_n=100):

        query = build_sql_ag_grid_from_params(start_date=start_date, end_date=end_date, tag_filter=tag_filter, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, compute_tag_keys=compute_tag_keys, top_n=top_n)
        session = self.system_query_manager.get_new_session()
        try:

            df = pd.read_sql(text(query), session.bind)
            df['rowIndex'] = df.index
        finally:
            session.close()
        return df #.to_dict('records')


    ##### Get Compute Tagged Table
    def get_compute_tagged_grid_data(self):

        ComputeTaggedPoliciesAGGrid = self.system_query_manager.reflect_table('app_compute_tags')
        session =self.system_query_manager.get_new_session()
        try:
            query = session.query(ComputeTaggedPoliciesAGGrid)
            df = pd.read_sql(query.statement, session.bind)
            df['rowIndex'] = df.index
        finally:
            session.close()
        return df #.to_dict('records')


    def get_base_tag_page_filter_defaults(self):

        system_engine = self.system_query_manager.get_engine()

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

        return df_date_min_filter, df_date_max_filter, current_date_filter, day_30_rolling_filter, df_product_cat_filter, df_cluster_id_filter, df_job_id_filter



    def get_tag_filters(self):
        system_engine = self.system_query_manager.get_engine()

        # Fetch distinct tag policy names within the context manager
        with self.system_query_manager.session_scope(system_engine) as session:

            ## 1 Query Instead of 3
            tag_policy_result = session.query(TagPolicies.tag_policy_name, TagPolicies.tag_key, TagPolicies.tag_value).all()

            tag_keys = session.query(text("""
                                        DISTINCT explode(map_keys(custom_tags)) AS TagKeys FROM clean_usage"""
                                        )).all()
            
            compute_tag_keys_filter = [{'label': name[0] if name[0] is not None else 'None', 'value': name[0] if name[0] is not None else 'None'} for name in tag_keys]

            # Process the query result and store the distinct values in variables in your Dash app
            distinct_tag_policy_names = set()
            distinct_tag_keys = set()
            distinct_tag_values = set()

            for row in tag_policy_result:
                distinct_tag_policy_names.add(row.tag_policy_name)
                distinct_tag_keys.add(row.tag_key)
                if row.tag_value is not None:
                    distinct_tag_values.add(row.tag_value)

            tag_policy_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_policy_names]
            tag_key_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_keys]
            tag_value_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_values]

        return compute_tag_keys_filter, tag_policy_filter, tag_key_filter, tag_value_filter




#### Tagging Advisor Page Function

def render_tagging_advisor_page(df_date_min_filter, df_date_max_filter,
                                current_date_filter = datetime.now().date(), 
                                day_30_rolling_filter = datetime.now().date() - timedelta(days=30),
                                df_product_cat_filter = None,
                                df_cluster_id_filter = None,
                                df_job_id_filter = None,
                                compute_tag_keys_filter=None, 
                                tag_policy_filter=None, 
                                tag_key_filter=None, 
                                tag_value_filter=None,
                                ## Initial Condition Data Frames
                                tag_policies_grid_df = None,
                                compute_tagged_grid_df = None,
                                adhoc_clusters_grid_df = None,
                                jobs_clusters_grid_df = None,
                                sql_clusters_grid_df = None,
                                DEFAULT_TOP_N = 100
                                ):

    cellStyle = {
        "styleConditions": [
        {"condition": "highlightEdits(params)", "style": {"color": "orange"}},
    ]
    }

    defaultColDef = {
    "valueSetter": {"function": "addEdits(params)"},
    "editable": True,
    "sortable": True,
    "filter": True,
    "cellStyle": cellStyle
    }


    ##### REDER TAGGING PAGE
    layout = dbc.Container([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H1("Databricks Tagging Manager", style={'color': '#002147'}),  # A specific shade of blue
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Label('Compute Tag Key Filter', style={'font-weight': 'bold', 'color': '#002147'}),
                        # RadioItems component for the filter
                            dcc.Dropdown(
                                id='compute-tag-filter-dropdown',
                                options=compute_tag_keys_filter,
                                value='All', 
                                multi=True,
                                clearable=True  # Prevents user from clearing the selection, ensuring a selection is always active
                            , style={'margin-bottom': '2px', 'margin-top': '10px'}),
                            # Output component to display the result based on the selected filter
                            html.Div(id='filter-output')
                    ])
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.Label('Tag Policy Match Filter', style={'font-weight': 'bold', 'color': '#002147'}),
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
                    html.Button('Refresh', id='update-params-button', n_clicks=0, className = 'prettier-button'),  # A specific shade of blue
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
                                children=html.Div([dcc.Graph(id='matched-usage-ind', className = 'chart-visuals', config={'responsive': True})]),
                                fullscreen=False,  # True to make the spinner fullscreen
                                color= '#002147')
                            ])
                ], width=3),
                dbc.Col([
                    html.Div([
                            dcc.Loading(
                                id="loading-unmatched-usage-chart",
                                type="default",  # Can be "graph", "cube", "circle", "dot", or "default"
                                children=html.Div([dcc.Graph(id='unmatched-usage-ind', className = 'chart-visuals', config={'responsive': True})]),
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
                        children=html.Div([dcc.Graph(id='percent-match-ind', className='chart-visuals', config={'responsive': True})]),
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
            ], id='output-data'),
            dbc.Row([dbc.Col([
                    html.Div([
                            dcc.Loading(
                                id="loading-tag-by-value-chart",
                                type="default",  # Can be "graph", "cube", "circle", "dot", or "default"
                                children=html.Div([dcc.Graph(id='usage-by-tag-value-line-chart', className = 'chart-visuals')]),
                                fullscreen=False,  # True to make the spinner fullscreen
                                color= '#002147')
                            ])
                        ], width=12)
            ]
            )
        ]
        ),

        html.Div(className='border-top'),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H2("Tag Audit Management", style={'color': '#002147'}),  # A specific shade of blue
                ], width=12)
            ]),
            html.Div(className='border-top'),
            dbc.Row([
                    dbc.Col([
                    html.P("This section allows users to create & manage tagging policies, find clusters/warehouses as well as jobs that are not properly tagged into the policies. \n Once these entities are found, they can then be properly categorized and tagging in the app.", style={'color': '#002147'}),  # A specific shade of blue
                ], width=12)
            ])
        ], style={'margin-left':'10px', 'margin-right':'10px'}),

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
                            ], width=3),
                            dbc.Col([
                                dcc.Loading(
                                    id="loading-clear-policies",
                                    type="default",  # Choose the style of the loading animation
                                    color= '#002147',
                                    children=html.Button('Clear Policy Changes', id='tag-policy-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                                )
                            ], width=3),
                                        # Spacer column to push the last two columns to the right
                            dbc.Col(
                                width={"size": 4}
                            ),
                            dbc.Col([
                                dcc.Loading(
                                    id="loading-add-policies",
                                    type="default",  # Choose the style of the loading animation
                                    color= '#002147',
                                    children=html.Button('+', id='add-policy-row-btn', n_clicks=0,
                                    className = 'agedit-button')
                                )
                            ], width=1),
                             dbc.Col([
                                dcc.Loading(
                                    id="loading-remove-policies",
                                    type="default",  # Choose the style of the loading animation
                                    color= '#002147',
                                    children=html.Button('-', id='remove-policy-row-btn', n_clicks=0, className = 'agedit-button')
                                )
                            ], width=1)
                        ]),
                        html.Div(id='policy-change-indicator', children=[
                                html.Span("! Pending Changes to be Saved", style={'color': '#DAA520'}),  # Dark yellow color
                            ], style={'display': 'none', 'margin-left':'10px', 'margin-right':'10px'}),
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
                                    'headerName': 'Policy Description', 
                                    'field': 'tag_policy_description', 
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
                                    'width': 150, 
                                    'enableCellChangeFlash': True,
                                    'suppressSizeToFit': True
                                },
                                {'headerCheckboxSelection': True, 'checkboxSelection': True, 'headerCheckboxSelectionFilteredOnly': True, 'width': 50, 
                                'suppressSizeToFit': True},
                            ],
                            defaultColDef=defaultColDef,
                            rowData=tag_policies_grid_df.to_dict('records'),
                            dashGridOptions={
                                'enableRangeSelection': True,
                                'rowSelection': 'multiple'
                                }
                        ),
                        dcc.Store(id='policy-changes-store')
                    ])
                    ], width=5, style={'margin-left':'10px', 'margin-right':'10px'}),

            ## AG Grid for active cluster / compute tags
                dbc.Col([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                    dcc.Loading(
                                    id="loading-save-compute-tags",
                                    type="default",  # Choose the style of the loading animation
                                    color= '#002147', 
                                    children=html.Button('Save Tag Changes', id='tag-compute-tags-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                                )
                            ], width=3),
                            dbc.Col([
                                dcc.Loading(
                                    id="loading-clear-compute-tags",
                                    type="default",  # Choose the style of the loading animation
                                    color= '#002147',
                                    children=html.Button('Clear Tag Changes', id='tag-compute-tags-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                                )
                            ], width=3),
                                        # Spacer to push the last two columns to the right
                            dbc.Col(
                                width={"size": 4}
                            ),
                            dbc.Col([
                                dcc.Loading(
                                    id="loading-add-compute-tags",
                                    type="default",  # Choose the style of the loading animation
                                    children=html.Button('+', id='add-compute-tag-row-btn', n_clicks=0,
                                    className = 'agedit-button')
                                )
                            ], width=1),
                             dbc.Col([
                                dcc.Loading(
                                    id="loading-remove-compute-tags",
                                    type="default",  # Choose the style of the loading animation
                                    children=html.Button('-', id='remove-compute-tags-row-btn', n_clicks=0,
                                    className = 'agedit-button')
                                )
                            ], width=1)
                        ]),
                        html.Div(id='compute-tag-change-indicator', children=[
                                html.Span("! Pending Changes to be Saved", style={'color': '#DAA520'}),  # Dark yellow color
                            ], style={'display': 'none', 'margin-left':'10px', 'margin-right':'10px'}),
                dag.AgGrid(
                    id='tag-compute-ag-grid',
                    columnDefs=[
                                {
                                    'headerName': 'Tag Id', 
                                    'field': 'tag_id', 
                                    'editable': False, 
                                    'width': 100, 
                                    'suppressSizeToFit': True
                                },
                                {
                                    'headerName': 'Compute Id', 
                                    'field': 'compute_asset_id', 
                                    'editable': True,
                                    'suppressSizeToFit': True
                                },
                                {
                                    'headerName': 'Compute Type', 
                                    'field': 'compute_asset_type', 
                                    'editable': True,
                                    'suppressSizeToFit': True
                                },
                                {
                                    'headerName': 'Tag Policy', 
                                    'field': 'tag_policy_name', 
                                    'width': 150, 
                                    'editable': True,
                                    'suppressSizeToFit': True
                                },
                                {
                                    'headerName': 'Tag Key', 
                                    'field': 'tag_key', 
                                    'editable': True,
                                    'width': 150, 
                                    'enableCellChangeFlash': True,
                                    'suppressSizeToFit': True
                                },
                                {
                                    'headerName': 'Tag Value', 
                                    'field': 'tag_value', 
                                    'editable': True,
                                    'width': 150, 
                                    'enableCellChangeFlash': True,
                                    'suppressSizeToFit': True
                                },
                                {
                                    'headerName': 'Is Persisted To Asset?', 
                                    'field': 'is_persisted_to_actual_asset', 
                                    'editable': True,
                                    'width': 80, 
                                    'enableCellChangeFlash': True,
                                    'suppressSizeToFit': True
                                },

                                {'headerCheckboxSelection': True, 'checkboxSelection': True, 'headerCheckboxSelectionFilteredOnly': True, 'width': 50, 
                                'suppressSizeToFit': True},
                            ],
                            defaultColDef=defaultColDef,
                            rowData= compute_tagged_grid_df.to_dict('records'),
                            dashGridOptions={
                                'enableRangeSelection': True,
                                'rowSelection': 'multiple'
                                }
                        ),
                        dcc.Store(id='cluster-tag-rowData-store', data=compute_tagged_grid_df.to_dict('records')),
                        dcc.Store(id='cluster-tag-changes-store', data=compute_tagged_grid_df.to_dict('records'))
                    ])
            ], width=6, style={'margin-left':'10px', 'margin-right':'10px', 'margin-bottom': '20px'})
        ]),
        
        ## DBC AG Grid Row 2,
        ### AG Grid for tagging adhoc usage
        ### Entirely New Row
        html.Div(className='border-top'),
        html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Adhoc Cluster Usage & Tags", style={'color': '#002147'}),  # A specific shade of blue
                        ], width=12, style={'margin-left':'10px', 'margin-right':'10px'})
                    ]),
                    html.Div(className='border-top'),
                    dbc.Row([
                            dbc.Col([
                            html.P("Tag Adhoc Clusters By Usage", style={'color': '#002147'}),  # A specific shade of blue
                        ], width=12, style={'margin-left':'10px', 'margin-right':'10px'})
                    ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="loading-save-adhoc-usage",
                                type="default",
                                color= '#002147', 
                                children=html.Button('Save Changes', id='usage-adhoc-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                            )
                        ], width=2),
                        dbc.Col([
                            dcc.Loading(
                                id="loading-clear-adhoc-usage",
                                type="default",
                                color= '#002147', 
                                children=html.Button('Clear Changes', id='usage-adhoc-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                            )
                        ], width=2),
                        dbc.Col(
                                width={"size": 6}
                            ),
                        dbc.Col([
                            html.P("Top N By Usage:", style={'color': '#002147', 'margin-bottom': '10px'}),
                            dcc.Input(
                                id='top-n-adhoc',
                                type='number',  # Set the input type to 'number'
                                placeholder='Top N Adhoc Clusters By Usage',
                                value = DEFAULT_TOP_N,
                                step=1,  # Step value to ensure only integers can be entered
                                debounce=True  # Wait for the user to finish typing before triggering the callback
                            ),
                        ], width=2, style={'margin-bottom': '10px'})
                    ]),
                    html.Div(id='usage-adhoc-change-indicator', children=[
                        html.Span("! Pending Changes to be Saved", style={'color': '#DAA520'}),  # Dark yellow color
                    ], style={'display': 'none'}),

                    dag.AgGrid(
                        id='adhoc-usage-ag-grid',
                        columnDefs=get_adhoc_ag_grid_column_defs(tag_key_filter),
                        defaultColDef=defaultColDef,
                        columnSize="sizeToFit",
                        rowData = adhoc_clusters_grid_df.to_dict('records'),#get_adhoc_clusters_grid_data(start_date=day_30_rolling_filter, end_date=current_date_filter, top_n=DEFAULT_TOP_N).to_dict('records'),  # Will be populated via callback
                        dashGridOptions={'enableRangeSelection': True, 'rowSelection': 'multiple'}
                    ),
                    dcc.Store(id='adhoc-ag-grid-store'),
                    dcc.Store(id='adhoc-ag-grid-original-data', data = adhoc_clusters_grid_df.to_dict('records')),
                    dcc.Store(id='tag-keys-store', data=tag_key_filter)
                ])
            ], width=11, style={'margin-left':'10px', 'margin-right':'10px', 'margin-bottom': '20px'})
        ]),
        ###### END Adhoc Usage Grid

        ## DBC AG Grid Row 2,
        ### AG Grid for tagging JOBS usage
        html.Div(className='border-top'),

        html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Jobs Usage & Tags", style={'color': '#002147'}),  # A specific shade of blue
                        ], width=12, style={'margin-left':'10px', 'margin-right':'10px'})
                    ]),
                    html.Div(className='border-top'),

                    dbc.Row([
                            dbc.Col([
                            html.P("Tag Jobs By Usage", style={'color': '#002147'}),  # A specific shade of blue
                        ], width=12, style={'margin-left':'10px', 'margin-right':'10px'})
                    ])
        ]),

        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="loading-save-jobs-usage",
                                type="default",
                                color= '#002147', 
                                children=html.Button('Save Changes', id='usage-jobs-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                            )
                        ], width=2),
                        dbc.Col([
                            dcc.Loading(
                                id="loading-clear-jobs-usage",
                                type="default",
                                color= '#002147',
                                children=html.Button('Clear Changes', id='usage-jobs-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                            )
                        ], width=2),
                        dbc.Col(
                                width={"size": 6}
                            ),
                        dbc.Col([
                            html.P("Top N By Usage:", style={'color': '#002147', 'margin-bottom': '10px'}),
                            dcc.Input(
                                id='top-n-jobs',
                                type='number',  # Set the input type to 'number'
                                placeholder='Top N Jobs Clusters By Usage',
                                value = DEFAULT_TOP_N,
                                step=1,  # Step value to ensure only integers can be entered
                                debounce=True  # Wait for the user to finish typing before triggering the callback
                            ),
                        ], width=2, style={'margin-bottom': '10px'})
                    ]),

                    html.Div(id='usage-jobs-change-indicator', children=[
                        html.Span("! Pending Changes to be Saved", style={'color': '#DAA520'}),  # Dark yellow color
                    ], style={'display': 'none'}),

                    dag.AgGrid(
                        id='jobs-usage-ag-grid',
                        columnDefs=[
                            {'headerCheckboxSelection': True, 'checkboxSelection': True, 'headerCheckboxSelectionFilteredOnly': True, 'width': 50, 'suppressSizeToFit': True},
                            {'headerName': 'Job Id', 'field': 'job_id', 'editable': False, 'width': 100, 'suppressSizeToFit': True},
                            {'headerName': 'Policy Status', 'field': 'is_tag_policy_match', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Missing Policy Tags', 'field': 'missing_tags', 'editable': True, 'suppressSizeToFit': True},
                            {'headerName': 'Tags', 'field': 'tags', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Add Policy Key', 'field': 'input_policy_key', 'editable': True, 'cellStyle': {'backgroundColor': 'rgba(111, 171, 208, 0.9)'}, 'suppressSizeToFit': True},
                            # 'cellEditor': 'agSelectCellEditor', 'cellEditorParams': {'values': [option.get('value') for option in tag_keys]}},
                            {'headerName': 'Add Policy Value', 'field': 'input_policy_value', 'editable': True, 'cellStyle': {'backgroundColor': 'rgba(111, 171, 208, 0.9)'},'suppressSizeToFit': True},
                            {'headerName': 'Usage Amount', 'field': 'Dollar_DBUs_List', 'editable': False, 'suppressSizeToFit': True,
                                    'cellRenderer': 'BarGuage', 'guageColor': 'rgba(111, 171, 208, 0.7)'},
                            {'headerName': 'T7 Usage', 'field': 'T7_Usage', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'BarGuage', 'guageColor':'rgba(172, 213, 180, 0.6)'},
                            {'headerName': 'T30 Usage', 'field': 'T30_Usage', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'BarGuage', 'guageColor':'rgba(172, 213, 180, 0.6)'},
                            {'headerName': 'T90 Usage', 'field': 'T90_Usage', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'BarGuage', 'guageColor':'rgba(172, 213, 180, 0.6)'},
                            {'headerName': 'Resource Age', 'field': 'resource_age', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'HeatMap'},
                            {'headerName': 'Days Since Last Use', 'field': 'days_since_last_use', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'HeatMap'},
                            {'headerName': 'First Usage Date', 'field': 'first_usage_date', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Latest Usage Date', 'field': 'latest_usage_date', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Product Type', 'field': 'product_type', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Workspace ID', 'field': 'workspace_id', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Account ID', 'field': 'account_id', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Resource Owner', 'field': 'resource_owner', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Usage Quantity', 'field': 'usage_quantity', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Cluster Name', 'field': 'cluster_name', 'editable': False, 'suppressSizeToFit': True}
                        ],
                        defaultColDef=defaultColDef,
                        columnSize="sizeToFit",
                        rowData = jobs_clusters_grid_df.to_dict('records'),#get_jobs_clusters_grid_data(start_date=day_30_rolling_filter, end_date=current_date_filter, top_n=DEFAULT_TOP_N).to_dict('records'),  # Will be populated via callback
                        dashGridOptions={'enableRangeSelection': True, 'rowSelection': 'multiple'}
                    ),
                    dcc.Store(id='jobs-ag-grid-store'),
                    dcc.Store(id='jobs-ag-grid-original-data', data = jobs_clusters_grid_df.to_dict('records'))
                ])
            ], width=11, style={'margin-left':'10px', 'margin-right':'10px', 'margin-bottom': '20px'})
        ]),

        ###### END JOBS Usage Grid
         ## DBC AG Grid Row 2,
        ### AG Grid for tagging JOBS usage
        html.Div(className='border-top'),
        html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H3("SQL Warehouse Usage & Tags", style={'color': '#002147'}),  # A specific shade of blue
                        ], width=12, style={'margin-left':'10px', 'margin-right':'10px'})
                    ]),
                    html.Div(className='border-top'),
                    dbc.Row([
                            dbc.Col([
                            html.P("Tag SQL Warehouse By Usage", style={'color': '#002147'}),  # A specific shade of blue
                        ], width=12, style={'margin-left':'10px', 'margin-right':'10px'})
                    ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="loading-save-sql-usage",
                                type="default",
                                color= '#002147', 
                                children=html.Button('Save Changes', id='usage-sql-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                            )
                        ], width=2),
                        dbc.Col([
                            dcc.Loading(
                                id="loading-clear-sql-usage",
                                type="default",
                                color= '#002147',
                                children=html.Button('Clear Changes', id='usage-sql-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
                            )
                        ], width=2),
                        dbc.Col(
                                width={"size": 6}
                            ),
                        dbc.Col([
                            html.P("Top N By Usage:", style={'color': '#002147', 'margin-bottom': '10px'}),
                            dcc.Input(
                                id='top-n-sql',
                                type='number',  # Set the input type to 'number'
                                placeholder='Top N SQL Warehouses By Usage',
                                value = DEFAULT_TOP_N,
                                step=1,  # Step value to ensure only integers can be entered
                                debounce=True  # Wait for the user to finish typing before triggering the callback
                            ),
                        ], width=2, style={'margin-bottom': '10px'})
                    ]),

                    html.Div(id='usage-sql-change-indicator', children=[
                        html.Span("! Pending Changes to be Saved", style={'color': '#DAA520'}),  # Dark yellow color
                    ], style={'display': 'none'}),

                    dag.AgGrid(
                        id='sql-usage-ag-grid',
                        columnDefs=[
                            {'headerCheckboxSelection': True, 'checkboxSelection': True, 'headerCheckboxSelectionFilteredOnly': True, 'width': 50, 'suppressSizeToFit': True},
                            {'headerName': 'Warehouse Id', 'field': 'warehouse_id', 'editable': False, 'width': 100, 'suppressSizeToFit': True},
                            {'headerName': 'Policy Status', 'field': 'is_tag_policy_match', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Missing Policy Tags', 'field': 'missing_tags', 'editable': True, 'suppressSizeToFit': True},
                            {'headerName': 'Tags', 'field': 'tags', 'editable': True, 'suppressSizeToFit': True},
                            {'headerName': 'Add Policy Key', 'field': 'input_policy_key', 'editable': True, 'cellStyle': {'backgroundColor': 'rgba(111, 171, 208, 0.9)'}, 'suppressSizeToFit': True},
                            # 'cellEditor': 'agSelectCellEditor', 'cellEditorParams': {'values': [option.get('value') for option in tag_keys]}},
                            {'headerName': 'Add Policy Value', 'field': 'input_policy_value', 'editable': True, 'cellStyle': {'backgroundColor': 'rgba(111, 171, 208, 0.9)'},'suppressSizeToFit': True},
                            {'headerName': 'Usage Amount', 'field': 'Dollar_DBUs_List', 'editable': False, 'suppressSizeToFit': True,
                                    'cellRenderer': 'BarGuage', 'guageColor': 'rgba(111, 171, 208, 0.7)'},
                            {'headerName': 'T7 Usage', 'field': 'T7_Usage', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'BarGuage', 'guageColor':'rgba(172, 213, 180, 0.6)'},
                            {'headerName': 'T30 Usage', 'field': 'T30_Usage', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'BarGuage', 'guageColor':'rgba(172, 213, 180, 0.6)'},
                            {'headerName': 'T90 Usage', 'field': 'T90_Usage', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'BarGuage', 'guageColor':'rgba(172, 213, 180, 0.6)'},
                            {'headerName': 'Resource Age', 'field': 'resource_age', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'HeatMap'},
                            {'headerName': 'Days Since Last Use', 'field': 'days_since_last_use', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'HeatMap'},
                            {'headerName': 'First Usage Date', 'field': 'first_usage_date', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Latest Usage Date', 'field': 'latest_usage_date', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Product Type', 'field': 'product_type', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Workspace ID', 'field': 'workspace_id', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Account ID', 'field': 'account_id', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Resource Owner', 'field': 'resource_owner', 'editable': False, 'suppressSizeToFit': True},
                            {'headerName': 'Usage Quantity', 'field': 'usage_quantity', 'editable': False, 'suppressSizeToFit': True}
                        ],
                        defaultColDef=defaultColDef,
                        columnSize="sizeToFit",
                        rowData = sql_clusters_grid_df.to_dict('records'),#get_sql_clusters_grid_data(start_date=day_30_rolling_filter, end_date=current_date_filter, top_n=DEFAULT_TOP_N).to_dict('records'),  # Will be populated via callback
                        dashGridOptions={'enableRangeSelection': True, 'rowSelection': 'multiple'}
                    ),
                    dcc.Store(id='sql-ag-grid-store'),
                    dcc.Store(id='sql-ag-grid-original-data', data = sql_clusters_grid_df.to_dict('records')) #get_sql_clusters_grid_data(start_date=day_30_rolling_filter, end_date=current_date_filter, top_n=DEFAULT_TOP_N).to_dict('records')),
                ])
            ], width=11, style={'margin-left':'10px', 'margin-right':'10px', 'margin-bottom': '20px'}),
            html.Div(id='save-adhoc-trigger', style={'display': 'none'}),
            html.Div(id='save-jobs-trigger', style={'display': 'none'}) ,
            html.Div(id='save-tags-trigger', style={'display': 'none'}) , ## From when we just need to reload locally - no DB changes
            html.Div(id='reload-tags-trigger', style={'display': 'none'}) , ## For when we actually need to reload from DB
            html.Div(id='save-sql-trigger', style={'display': 'none'}) ### Output element so all AG grids can update the compute tags to one output callback
        ]),
        ###### END SQL Usage Grid

    dbc.Container(id='output-container')
    ], fluid=True, style={'width': '83.3vw'})

    return layout



