
"""
Author: Cody Austin Davis
Date: 5/25/2024
Description: Dash Apps for Tagging and Data Compute Observability

TO DO: 
1. Create settings page that lets a user input the workspace name and select the warehouse to execute against. (OAuth)

7. Create LLM to generate visuals for system tables upon request / Alerts
9. Need to add Warehouse Name and owner once Warehouses system tables is available
10. Handle partial saves for alerts/queries/jobs so that you dont create dups

BUGS: 
1. Make filters update based on other selected filter values
2. Better cache handling
"""


import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from databricks import sql
import os
from dotenv import load_dotenv
from page_functions import *
from data_functions import *
from chart_functions import ChartFormats
from agent_functions import (create_alert_and_job, delete_alert_and_job, parse_query_result_json_from_string)
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sqlalchemy import bindparam
from threading import Thread
from data_functions.utils import *
from pandasql import sqldf
from flask_caching import Cache
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql, jobs
from datetime import date, datetime, time, timedelta, timezone


## Log SQL Alchemy Commands
logging.basicConfig(level=logging.ERROR)
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

## Initialize Database

# Load environment variables from .env file
# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, './', 'config', '.env')
# Load the environment variables
load_dotenv(dotenv_path=env_path)

## load auth into variables
host = os.getenv("DATABRICKS_SERVER_HOSTNAME")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
access_token = os.getenv("DATABRICKS_TOKEN")
catalog = os.getenv("DATABRICKS_CATALOG")
schema = os.getenv("DATABRICKS_SCHEMA")

## Get Warehouse Id from http_path
last_slash_index = http_path.rfind('/')
warehouse_id = http_path[last_slash_index + 1:] if last_slash_index != -1 else http_path


## Load Visual Specific Query Parts
AGG_QUERY = read_sql_file("./config/tagging_advisor/tag_date_agg_query.sql")
MATCHED_IND_QUERY = read_sql_file("./config/tagging_advisor/matched_indicator_query.sql")
TAG_VALUES_QUERY = read_sql_file("./config/tagging_advisor/tag_values_query.sql")
HEATMAP_QUERY = read_sql_file("./config/tagging_advisor/tag_sku_heatmap_query.sql")
TAG_VALUES_OVER_TIME_QUERY = read_sql_file("./config/tagging_advisor/tag_values_over_time.sql")
DEFAULT_TOP_N = 100

### Client for interacting with all of Databricks outside of submitting SQL commands
dbx_client = WorkspaceClient(
        host=host,
        token = access_token
    )

###########  Initialize the Dash app with a Bootstrap theme ###########
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

##### Load Init Scripts 
## Create Engine on start up
### System query manager uses the system catalog and schema scope (for the tables it creates and manages)
system_query_manager = QueryManager(host=host, http_path=http_path, access_token= access_token, catalog= catalog, schema = schema)
system_engine = system_query_manager.get_engine()

### Functions to process data for these tabs
tag_advisor_manager = TagAdvisorPageManager(system_query_manager=system_query_manager)
setting_manager = SettingsPageManager(system_query_manager=system_query_manager)
alerts_manager = AlertsManager(system_query_manager=system_query_manager)


##### Initialize Cache for Data Functions
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': './config/cache/'
})

#### Define cached data calling functions ######
@cache.memoize(timeout=600)  # Cache the results for 5 minutes
def cached_execute_query_to_df(query):
    return system_query_manager.execute_query_to_df(query)

### Cache the start up load specifically
@cache.memoize(timeout=600)
def cached_get_base_tag_page_filter_defaults():
    return tag_advisor_manager.get_base_tag_page_filter_defaults()

@cache.memoize(timeout=600)
def cached_get_tag_filters():
    return tag_advisor_manager.get_tag_filters()

@cache.memoize(timeout=600)
def cached_get_tag_policies_grid_data():
    return tag_advisor_manager.get_tag_policies_grid_data()

@cache.memoize(timeout=60)
def cached_get_compute_tagged_grid_data():
    return tag_advisor_manager.get_compute_tagged_grid_data()

@cache.memoize(timeout=60)
def cached_get_adhoc_clusters_grid_data(start_date, end_date, tag_filter=None, tag_policies=None, product_category=None, tag_keys=None, tag_values=None, compute_tag_keys=None, top_n=100):
    return tag_advisor_manager.get_adhoc_clusters_grid_data(start_date = start_date, end_date = end_date, tag_filter=tag_filter, tag_policies=tag_policies, product_category=product_category, tag_keys=tag_keys, tag_values=tag_values, compute_tag_keys=compute_tag_keys, top_n=top_n)


@cache.memoize(timeout=60)
def cached_get_jobs_clusters_grid_data(start_date, end_date, tag_filter=None, tag_policies=None, product_category=None, tag_keys=None, tag_values=None, compute_tag_keys=None, top_n=100):
    return tag_advisor_manager.get_jobs_clusters_grid_data(start_date = start_date, end_date = end_date, tag_filter=tag_filter, tag_policies=tag_policies, product_category=product_category, tag_keys=tag_keys, tag_values=tag_values, compute_tag_keys=compute_tag_keys, top_n=top_n)


@cache.memoize(timeout=60)
def cached_get_sql_clusters_grid_data(start_date, end_date, tag_filter=None, tag_policies=None, product_category=None, tag_keys=None, tag_values=None, compute_tag_keys=None, top_n=100):
    return tag_advisor_manager.get_sql_clusters_grid_data(start_date = start_date, end_date = end_date, tag_filter=tag_filter, tag_policies=tag_policies, product_category=product_category, tag_keys=tag_keys, tag_values=tag_values, compute_tag_keys=compute_tag_keys, top_n=top_n)



####### Load Defaults for App Start Up
## Create Data Model and MV App Is Based On
tag_advisor_manager.run_init_scripts()


### Init Load

#tag_advisor_manager.get_base_tag_page_filter_defaults()
df_date_min_filter, df_date_max_filter, current_date_filter, day_30_rolling_filter, df_product_cat_filter, df_cluster_id_filter, df_job_id_filter = cached_get_base_tag_page_filter_defaults()
#tag_advisor_manager.get_tag_filters()
compute_tag_keys_filter, tag_policy_filter, tag_key_filter, tag_value_filter = cached_get_tag_filters()

## Tag Poligies AG Grid DF
# tag_advisor_manager.get_tag_policies_grid_data()
## Do not cache these, they are live edits
tag_policies_grid_df = tag_advisor_manager.get_tag_policies_grid_data()
#tag_advisor_manager.get_compute_tagged_grid_data()
compute_tagged_grid_df = tag_advisor_manager.get_compute_tagged_grid_data()
alerts_ag_grid_data = alerts_manager.get_alerts_ag_grid_data()

##
adhoc_clusters_grid_df = cached_get_adhoc_clusters_grid_data(start_date=day_30_rolling_filter, end_date=current_date_filter, top_n=DEFAULT_TOP_N)
jobs_clusters_grid_df = cached_get_jobs_clusters_grid_data(start_date=day_30_rolling_filter, end_date=current_date_filter, top_n=DEFAULT_TOP_N)
sql_clusters_grid_df = cached_get_sql_clusters_grid_data(start_date=day_30_rolling_filter, end_date=current_date_filter, top_n=DEFAULT_TOP_N)

#### Load Defaults for Tag Manager
### TO DO: - Cache these bois


# Define the layout of the main app

app.layout = dbc.Container([
    dcc.Store(id='sidebar-state', data={'is_open': True}),
    dbc.Row([
        dbc.Col([
                dbc.Container(
                    dbc.Nav(
                    [
                        dbc.Button("☰", id="toggle-button", n_clicks=0, className='toggle-button'),
                        html.Div([  # Container for the logo and possibly other header elements
                            html.Img(id='sidebar-logo', src='/assets/app_logo.png', style={'width': '100%', 'height': 'auto', 'padding': '0px'}),
                        ]),
                        dbc.NavLink("Tags", href="/tag-manager", id="tab-1-link", active='exact'),
                        dbc.NavLink("Alerts", href="/alert-manager", id="tab-2-link", active='exact'),
                        dbc.NavLink("Contracts", href="/contract-manager", id="tab-3-link", active='exact'),
                        dbc.NavLink("Settings", href="/settings", id="tab-4-link", active='exact'),
                    ],
                    vertical=True,
                    pills=True,
                    className="sidebar",
                    id='sidebar'
                ), fluid=True)
        ], width=2, id="sidebar-col", className="sidebar-col"),
        dbc.Col([
            dcc.Location(id='url', refresh=False),
            dbc.Container(id='tab-content', className="main-content", fluid=True)
        ], id="main-content-col", width=10)
    ])
], className="app-container", fluid=True)



### Manage Tab Selections
@app.callback(
    [Output('tab-1-link', 'active'),
     Output('tab-2-link', 'active'),
     Output('tab-3-link', 'active'),
     Output('tab-4-link', 'active')],
    [Input('url', 'pathname')]
)
def set_active_tab(pathname):
    if pathname == "/":
        # Assuming the default path should activate the first tab
        return True, False, False, False
    return pathname == "/tag-manager", pathname == "/alert-manager", pathname == "/contract-manager", pathname == "/settings", 



@app.callback(
    [
     Output("sidebar-col", "width"),
     Output("main-content-col", "width"),
     Output("toggle-button", "children"),
     Output("sidebar-state", "data")],
    [Input("toggle-button", "n_clicks")],
    [State("sidebar-state", "data")]
)
def toggle_sidebar(n, sidebar_state):
    if n:
        is_open = not sidebar_state['is_open']
        sidebar_state['is_open'] = is_open
        sidebar_width = 2 if is_open else 1
        main_content_width = 10 if is_open else 11
        button_text = "☰" if is_open else "☰"
        logo_style = {'width': '100%', 'height': 'auto'} #if is_open else {'display': 'none'}
        return sidebar_width, main_content_width, button_text, sidebar_state
    return 2, 10, "☰" , sidebar_state



# Callback to update the content based on tab selection
@app.callback(
    Output('tab-content', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):

    if pathname == "/tag-manager":

        ## TO DO: Add authentication screen
        ## Can one of these be called from an LLM to create visuals???

        ## Load AG Grids with Initial Conditions
        return  render_tagging_advisor_page(df_date_min_filter, df_date_max_filter,
                                current_date_filter = current_date_filter, 
                                day_30_rolling_filter = day_30_rolling_filter,
                                df_product_cat_filter = df_product_cat_filter,
                                df_cluster_id_filter = df_cluster_id_filter,
                                df_job_id_filter = df_job_id_filter,
                                compute_tag_keys_filter= compute_tag_keys_filter, 
                                tag_policy_filter=tag_policy_filter, 
                                tag_key_filter=tag_key_filter, 
                                tag_value_filter=tag_value_filter,
                                ## Initial Condition Data Frames
                                tag_policies_grid_df = tag_policies_grid_df,
                                compute_tagged_grid_df = compute_tagged_grid_df,
                                adhoc_clusters_grid_df = adhoc_clusters_grid_df,
                                jobs_clusters_grid_df = jobs_clusters_grid_df,
                                sql_clusters_grid_df = sql_clusters_grid_df,
                                DEFAULT_TOP_N = DEFAULT_TOP_N
                                )
    
    elif pathname == "/alert-manager":
        return  render_alert_manager_page(alerts_ag_grid_data = alerts_ag_grid_data)
    elif pathname == "/contract-manager":
        return render_contract_manager_page()
    elif pathname == "/settings":
        return render_settings_page()
    else:
        return render_tagging_advisor_page(df_date_min_filter, df_date_max_filter,
                                current_date_filter = current_date_filter, 
                                day_30_rolling_filter = day_30_rolling_filter,
                                df_product_cat_filter = df_product_cat_filter,
                                df_cluster_id_filter = df_cluster_id_filter,
                                df_job_id_filter = df_job_id_filter,
                                compute_tag_keys_filter= compute_tag_keys_filter, 
                                tag_policy_filter=tag_policy_filter, 
                                tag_key_filter=tag_key_filter, 
                                tag_value_filter=tag_value_filter,
                                ## Initial Condition Data Frames
                                tag_policies_grid_df = tag_policies_grid_df,
                                compute_tagged_grid_df = compute_tagged_grid_df,
                                adhoc_clusters_grid_df = adhoc_clusters_grid_df,
                                jobs_clusters_grid_df = jobs_clusters_grid_df,
                                sql_clusters_grid_df = sql_clusters_grid_df,
                                DEFAULT_TOP_N = DEFAULT_TOP_N
                                )


@app.callback(
    Output("refresh-mv-output", "children"),
    Output("update-schedule-output", "children"),
    Input("set-mv-schedule", "n_clicks"),
    Input("refresh-mv", "n_clicks"),
    State("mv-cron-schedule", "value"),
    prevent_initial_call=True
)
def handle_mv_actions(set_mv_clicks, refresh_mv_clicks, cron_schedule):
    ctx = dash.callback_context

    if not ctx.triggered:
        # No button has been clicked yet
        return "", ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "set-mv-schedule" and cron_schedule:
        # Setting the materialized view schedule
        result = setting_manager.set_materialized_view_schedule(cron_schedule=cron_schedule)
        return "", result 

    elif button_id == "refresh-mv":
        # Refreshing the materialized view
        result = setting_manager.refresh_materialized_view()
        return result, "" 

    return "", "" 

@app.callback(
    Output("cache-clear-output", "children"),
    Input("clear-cache-button", "n_clicks"),
    prevent_initial_call=True  # Prevents the callback from running upon loading the app
)
def clear_cache(n_clicks):
    if n_clicks:
        cache.clear()
        return "Cache has been cleared successfully!"
    return ""  # This will not actually run because of `prevent_initial_call`

##### Update Usage Chart


@app.callback(
    Output('usage-by-match-chart', 'figure'),
    Output('matched-usage-ind', 'figure'),
    Output('unmatched-usage-ind', 'figure'),
    Output('usage-by-tag-value-chart', 'figure'),
    Output('usage-heatmap', 'figure'),
    Output('percent-match-ind', 'figure'),
    Output('total-usage-ind', 'figure'),
    Output('usage-by-tag-value-line-chart', 'figure'),
    Input('update-params-button', 'n_clicks'),
    State('tag-filter-dropdown', 'value'),
    State('start-date-picker', 'date'),
    State('end-date-picker', 'date'),
    State('product-category-dropdown', 'value'), ## Tuple
    State('tag-policy-dropdown', 'value'), ## Tuple
    State('tag-policy-key-dropdown', 'value'), ## Tuple
    State('tag-policy-value-dropdown', 'value'), ## Tuple
    State('compute-tag-filter-dropdown', 'value') ## Tuple
)
def update_usage_by_match(n_clicks, tag_filter, start_date, end_date, product_category, tag_policies, tag_keys, tag_values, compute_tag_keys):


        
    # Define the color mapping for compliance
    color_map = {
        'In Policy': '#097969',
        'Not Matched To Tag Policy': '#002147'
    }

    # Convert the result to a DataFrame
    usage_by_match_bar_query = build_tag_query_from_params(tag_filter=tag_filter, start_date=start_date, end_date=end_date, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, compute_tag_keys=compute_tag_keys, final_agg_query=AGG_QUERY)

    # Convert the result to a DataFrame
    usage_by_match_bar_df = cached_execute_query_to_df(usage_by_match_bar_query)

    # Create a Plotly Express line chart
    fig = px.bar(usage_by_match_bar_df, x='Usage_Date', y='Usage Amount', 
                    title= 'Daily Usage By Tag Policy Match',
                    color='Tag Match',
                    labels={'Usage Amount': 'Usage Amount', 'Usage_Date': 'Usage Date'},
                    barmode="group",
                    #bargap=0.15,
                    #bargroupgap=0.1,
                    color_discrete_map=color_map)
    
    fig.update_layout(ChartFormats.common_chart_layout())
    fig.update_layout(
            legend=dict(
                x=0,
                y=-0.3,  # You may need to adjust this depending on your specific plot configuration
                orientation="h"  # Horizontal legend
            )
        )
    

    #### Matched Usage Indicator
    ind_query = build_tag_query_from_params(tag_filter=tag_filter, start_date=start_date, end_date=end_date, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, compute_tag_keys=compute_tag_keys, final_agg_query=MATCHED_IND_QUERY)
    ind_df = cached_execute_query_to_df(ind_query)
    matched_value = safe_round(ind_df['Matched Usage Amount'][0], 0)
    not_matched_value = safe_round(ind_df['Not Matched Usage Amount'][0], 0)


    #### Percent Matched Usage Indicator
    percent_match_fig = go.Figure(go.Indicator(
                            mode="number",
                            value=safe_divide(matched_value, safe_add(matched_value, not_matched_value)) ,
                            title={"text": "% Matched Usage", 'font': {'size': 24}},
                            number={'font': {'size': 42, 'color': "#097969"}, 'valueformat': ',.1%'}
                            ))
    
    percent_match_fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=10)
    )


    #### Total Usage Indicator
    total_usage_fig = go.Figure(go.Indicator(
                            mode="number",
                            value=safe_round(safe_add(matched_value, not_matched_value), 1) ,
                            title={"text": "Total Usage", 'font': {'size': 24}},
                            number={'font': {'size': 42, 'color': "#097969"}, 'valueformat': '$,'}
                            ))
    
    total_usage_fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    #### Matched Usage Amount Indicator Guage
    matched_fig = go.Figure(go.Indicator(
                            mode="number+gauge",
                            value=matched_value,
                            title={"text": "Matched Usage", 'font': {'size': 24}},
                            number={'font': {'size': 24, 'color': "#097969"}, 'valueformat': '$,'},
                            gauge={'shape': "angular",
                            'axis': {'range': [0, matched_value + not_matched_value]},
                            'bar': {'color': "#097969"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#002147"}
                            ))
    
    matched_fig.update_layout(
        height=180,
        autosize=True,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    #### Not Matched Usage Indicator
    unmatched_fig = go.Figure(go.Indicator(
                            mode="number+gauge",
                            value=not_matched_value,
                            title={"text": "Not Matched Usage", 'font': {'size': 24}, 'align': 'center'},
                            number={'font': {'size': 24, 'color': '#8B0000'}, 'valueformat': '$,'},
                            gauge={'shape': "angular",
                            'axis': {'range': [0, matched_value + not_matched_value]},
                            'bar': {'color': '#8B0000'},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#002147"}
                            ))
    
    unmatched_fig.update_layout(
        height=180,
        autosize=True,
        margin=dict(l=10, r=10, t=60, b=10)
    )


    #### Usage By Tag Value Bar Chart
    values_query_over_time = build_tag_query_from_params(
                                            tag_filter=tag_filter, start_date=start_date, end_date=end_date, 
                                            product_category=product_category, 
                                            tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, 
                                            compute_tag_keys=compute_tag_keys,
                                            final_agg_query=TAG_VALUES_OVER_TIME_QUERY)
    
    values_over_time_df = cached_execute_query_to_df(values_query_over_time)

    ## 2 queries in 1, just sort on client side, lessen load on warehouse

    values_df = values_over_time_df.groupby('Tag Value In Policy')['Usage Amount'].sum().reset_index().sort_values(by='Usage Amount', ascending=True)

    #print(values_df)
    ##system_query_manager.execute_query_to_df(values_query).sort_values(by='Usage Amount', ascending=True)

    ## Color Gradient

    norm = np.linspace(0, 1, len(values_df))
    colors_gradient = [
    f"rgb({int(182 - 182 * x)}, {int(217 - 143 * x)}, {int(168 - 56 * x)})"
    for x in norm
    ]

    tag_values_bar = go.Figure(
        go.Bar(
            y=values_df['Tag Value In Policy'],
            x=values_df['Usage Amount'],
            orientation='h',
            #text_auto=True,
            marker_color=colors_gradient
        )
    )
    tag_values_bar.update_layout(ChartFormats.common_chart_layout())

    # Update layout for a better look
    tag_values_bar.update_layout(
    title='Usage by Tag Value',
    xaxis_title='Total Usage',
    yaxis_title='Tag Policy Value'
    )
    tag_values_bar.update_layout(xaxis_type = "log")


    ##### Usage By Values Over Time Query
    
    # Define the color scale
    tag_values_color_scale = px.colors.qualitative.Plotly[:len(values_over_time_df['Tag Value In Policy'].unique())]
    # Find the index of the category 'B' and replace its corresponding color with grey


    # Find the index of the category 'B' if it exists
    category = 'Not In Policy'
    if category in values_over_time_df['Tag Value In Policy'].unique():
        tag_value_category_index = values_over_time_df['Tag Value In Policy'].unique().tolist().index('Not In Policy')
        if tag_value_category_index < len(tag_values_color_scale):
            tag_values_color_scale[tag_value_category_index] = 'grey'


    values_over_time_fig = px.bar(
        values_over_time_df,
        x='Usage Date',
        y='Usage Amount',
        title='Usage Over Time By Tag Value',
        color='Tag Value In Policy',
        labels={'Usage Amount': 'Usage Amount', 'Usage_Date': 'Usage Date'},
        barmode='stack',  # Stacked bar chart
        #color_discrete_map=tag_values_color_scale,
        color_continuous_scale='Ocean'
    )

    
    values_over_time_fig.update_layout(ChartFormats.common_chart_layout())
    values_over_time_fig.update_layout(
            legend=dict(
                x=0,
                y=-0.3,  # You may need to adjust this depending on your specific plot configuration
                orientation="h"  # Horizontal legend
            )
        )



    #### Tags By SKU Heatmap
    heat_map_query = build_tag_query_from_params(
                                            tag_filter=tag_filter, start_date=start_date, end_date=end_date, 
                                            product_category=product_category, 
                                            tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, 
                                            compute_tag_keys=compute_tag_keys,
                                            final_agg_query=HEATMAP_QUERY)
    
    heat_map_df = cached_execute_query_to_df(heat_map_query).sort_values(by='Usage Amount', ascending=True) 
    heat_pivot = heat_map_df.pivot_table(values='Usage Amount', index='Tag', columns='Product', fill_value=0)
    #heat_pivot['Total'] = heat_pivot.sum(axis=1)

    # Define the number of steps in the gradient
    norm_heat = np.linspace(0, 1, len(heat_pivot))
    heat_colors_gradient = [
        [0, "rgba(0,0,0,0)"],  # Make 0 values transparent
        [0.01, "rgb(182, 217, 168)"]  # Start of actual color scale just above 0
    ]

    # Adding rest of the gradient
    heat_colors_gradient += [
        [x, f"rgb({int(182 - 182 * x)}, {int(217 - 143 * x)}, {int(168 - 56 * x)})"]
        for x in np.linspace(0.01, 1, len(heat_pivot))
    ]

    heat_map_fig = go.Figure(data=go.Heatmap(
                z=heat_pivot.values,  # Provide the values from the pivot table
                x=heat_pivot.columns,  # Set the x-axis to Product names
                y=heat_pivot.index,    # Set the y-axis to Tag names
                colorscale=heat_colors_gradient,
                hoverongaps = False))


    heat_map_fig.update_layout(
        title='Heatmap of Usage Quantity by Product and Tag',
        xaxis_title='Product',
        yaxis_title='Tag'
    )

    heat_map_fig.update_layout(ChartFormats.common_chart_layout())

    return fig, matched_fig, unmatched_fig, tag_values_bar, heat_map_fig, percent_match_fig, total_usage_fig, values_over_time_fig

##### AG Grid Callbacks

## Tag policy Management
## Incrementally Store and Manage Changes to the tag policy AG Grid
## Clear and Reload saved policies

@app.callback(
    [Output('policy-changes-store', 'data'),
     Output('tag-policy-ag-grid', 'rowData'),
     Output('policy-change-indicator', 'style'),
     Output('loading-save-policies', 'children'),
     Output('loading-clear-policies', 'children'),
     Output('tag-policy-dropdown', 'options'),
     Output('tag-policy-key-dropdown', 'options'),
     Output('tag-policy-value-dropdown', 'options'),
     Output('adhoc-usage-ag-grid', 'columnDefs'),
     Output('tag-keys-store', 'data')],
    [Input('tag-policy-save-btn', 'n_clicks'), 
     Input('tag-policy-clear-btn', 'n_clicks'),
     Input('tag-policy-ag-grid', 'cellValueChanged'),
     Input('add-policy-row-btn', 'n_clicks'),
     Input('remove-policy-row-btn', 'n_clicks')], 
    [State('tag-policy-ag-grid', 'rowData'),
     State('policy-changes-store', 'data'),
     State('tag-policy-ag-grid', 'selectedRows')]  # Current state of stored changes
)
def handle_policy_changes(save_clicks, clear_clicks, cell_change, add_row_clicks, remove_row_clicks, row_data, changes, selected_rows):

    ### Synchronously figure out what action is happening and run the approprite logic. 
    ## This single-callback method ensure that no strange operation order can happen. Only one can happen at once. 
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]


    ##### CREATE - Add New row in GRID
    if triggered_id == 'add-policy-row-btn' and add_row_clicks > 0:
        new_row = {
            'tag_policy_id': None,  # Will be generated by the database
            'tag_policy_name': '',
            'tag_key': '',
            'tag_value': '',
            'update_timestamp': datetime.now()
        }
        row_data.append(new_row)
        return dash.no_update, row_data, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    #### DELETE Handle removing selected rows
    
    if triggered_id == 'remove-policy-row-btn' and remove_row_clicks > 0:
        ## Only attempt to delete from

        ## TO DO: Be able to remove / delete rows by Row Id as well
        ids_to_remove = [row['tag_policy_id'] for row in selected_rows if row['tag_policy_id'] is not None]
        updated_row_data = [row for row in row_data if row['tag_policy_id'] not in ids_to_remove]
        
        if ids_to_remove:
            connection = system_query_manager.get_engine().connect()
            try:
                delete_query = text("""
                    DELETE FROM app_tag_policies
                    WHERE tag_policy_id IN :ids
                """).bindparams(bindparam('ids', expanding=True))
                connection.execute(delete_query, parameters= {'ids':ids_to_remove})
                connection.commit()

            except Exception as e:
                print(f"Error during deletion: {e}")
                raise e
            finally:
                connection.close()

        if changes is not None:
            updated_changes = [change for change in changes if change['tag_policy_id'] not in ids_to_remove]
        else:
            updated_changes = []

        # Fetch distinct tag policy names within the context manager
        with QueryManager.session_scope(system_engine) as session:

            ## 1 Query Instead of 3
            tag_policy_result = session.query(TagPolicies.tag_policy_name, TagPolicies.tag_key, TagPolicies.tag_value).all()

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


        return updated_row_data, tag_advisor_manager.get_tag_policies_grid_data().to_dict('records'), dash.no_update, dash.no_update, dash.no_update, tag_policy_filter, tag_key_filter, tag_value_filter, tag_advisor_manager.get_adhoc_ag_grid_column_defs(tag_key_filter), tag_key_filter

    ##### Handle Clear Button Press
    elif triggered_id == 'tag-policy-clear-btn' and clear_clicks:
        clear_loading_content = html.Button('Clear Policy Changes', id='tag-policy-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
        return [], tag_advisor_manager.get_tag_policies_grid_data().to_dict('records'), {'display': 'none'}, dash.no_update, clear_loading_content, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update # Clear changes and reload data
    
    
    ##### Handle cell change
    elif triggered_id == 'tag-policy-ag-grid' and cell_change:
        if changes is None:
            changes = []
        change_data = cell_change[0]['data']
        row_index = cell_change[0]['rowIndex']
        # Ensure the change data includes the row index
        change_data['rowIndex'] = row_index
        changes.append(change_data)
        row_data = mark_changed_rows(row_data, changes, row_id='tag_policy_id')

        return changes, row_data, {'display': 'block', 'color': 'yellow', 'font-weight': 'bold'}, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    ##### SAVE CHANGES

    # Handle saving changes
    if triggered_id == 'tag-policy-save-btn' and save_clicks:

        # Combine changes by row index
        grouped_changes = []

        if changes:
            grouped_changes = group_changes_by_row(changes) ## from data_functions.utils import *


        connection = system_query_manager.get_engine().connect()
        save_loading_content = html.Button('Save Policy Changes', id='tag-policy-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className='prettier-button')

        if changes:
            try:
                # Process grouped changes for both updates and inserts
                for change in grouped_changes:
                    #print("Combined change data:", change)  # Debug statement
                    record_id = change.get('tag_policy_id')
                    if record_id:
                        # Update existing record
                        update_query = text("""
                            UPDATE app_tag_policies
                            SET tag_policy_name = :tag_policy_name,
                                tag_policy_description = :tag_policy_description,
                                tag_key = :tag_key,
                                tag_value = :tag_value,
                                update_timestamp = NOW()
                            WHERE tag_policy_id = :tag_policy_id
                        """)
                        connection.execute(update_query, parameters={
                            'tag_policy_name': change['tag_policy_name'],
                            'tag_policy_description': change['tag_policy_description'],
                            'tag_key': change['tag_key'],
                            'tag_value': change['tag_value'],
                            'tag_policy_id': record_id
                        })

                        connection.commit()

                    else:
                        # Insert new record
                        if not change.get('tag_policy_name') or not change.get('tag_key'):
                            raise ValueError("Missing required fields: 'tag_policy_name' or 'tag_key'")

                        insert_params = {k: v for k, v in change.items() if k in ['tag_policy_name', 'tag_policy_description', 'tag_key', 'tag_value']}
                        print(f"INSERT PARAMS: {insert_params}")  # Debug statement

                        insert_query = text("""
                            INSERT INTO app_tag_policies (tag_policy_name, tag_policy_description, tag_key, tag_value, update_timestamp)
                            VALUES (:tag_policy_name, :tag_policy_description, :tag_key, :tag_value, NOW())
                        """)
                        connection.execute(insert_query,
                             parameters= {'tag_policy_name':insert_params['tag_policy_name'],
                                        'tag_policy_description':insert_params['tag_policy_description'],
                                        'tag_key':insert_params['tag_key'],
                                        'tag_value': insert_params['tag_value']})
                        
                        connection.commit()


            except Exception as e:
                print(f"Error during save with changes: {changes}")  # Debug error
                raise e
            finally:
                connection.close()

        else:
            pass
            
        with QueryManager.session_scope(system_engine) as session:

            ## 1 Query Instead of 3
            tag_policy_result = session.query(TagPolicies.tag_policy_name, TagPolicies.tag_key, TagPolicies.tag_value).all()

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

            return [], tag_advisor_manager.get_tag_policies_grid_data().to_dict('records'), {'display': 'none'}, save_loading_content, dash.no_update, tag_policy_filter, tag_key_filter, tag_value_filter, get_adhoc_ag_grid_column_defs(tag_key_filter), tag_key_filter


    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update  # No action taken


######################################################


#### Adhoc AG Grid Callback
@app.callback(
    [Output('adhoc-ag-grid-store', 'data'),
     Output('adhoc-usage-ag-grid', 'rowData'),
     Output('usage-adhoc-change-indicator', 'style'),
     Output('loading-save-adhoc-usage', 'children'),
     Output('loading-clear-adhoc-usage', 'children'),
     Output('save-adhoc-trigger', 'children')],
    ## Also need to update all filters and visuals? Or let the button to do? with the "Refresh Button"
    Input('update-params-button', 'n_clicks'),
    State('start-date-picker', 'date'),
    State('end-date-picker', 'date'),
    State('tag-filter-dropdown', 'value'),
    State('product-category-dropdown', 'value'), ## Tuple
    State('tag-policy-dropdown', 'value'), ## Tuple
    State('tag-policy-key-dropdown', 'value'), ## Tuple
    State('tag-policy-value-dropdown', 'value'), ## Tuple
    State('compute-tag-filter-dropdown', 'value'), ## Tuple
    ### Actual AG Grid State
    Input('usage-adhoc-save-btn', 'n_clicks'), 
    Input('usage-adhoc-clear-btn', 'n_clicks'),
    Input('adhoc-usage-ag-grid', 'cellValueChanged'),
    Input('top-n-adhoc', 'value'),
    State('adhoc-usage-ag-grid', 'rowData'),
     State('adhoc-ag-grid-store', 'data'),
     State('adhoc-ag-grid-original-data', 'data'),
     State('adhoc-usage-ag-grid', 'selectedRows')
)
def update_adhoc_grid_data(n_clicks, start_date, end_date, tag_filter, product_category, tag_policies, tag_keys, tag_values, compute_tag_keys,
                           save_clicks, clear_clicks, cell_change, top_n,
                           row_data, changes, original_data, selected_rows):

    
    ### Synchronously figure out what action is happening and run the approprite logic. 
    ### This single-callback method ensure that no strange operation order can happen. Only one can happen at once. 
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]


    ##### Handle Changes to top N filter
    #### This is what is happenening
    ### If the refresh button gets clicked, refresh the AG Grids
    if (triggered_id == 'top-n-adhoc' and top_n) or (triggered_id == 'update-params-button' and n_clicks > 0):
       
       updated_data = tag_advisor_manager.get_adhoc_clusters_grid_data(
        start_date=start_date,
        end_date=end_date,
        tag_filter=tag_filter,
        product_category = product_category,
        tag_policies=tag_policies,
        tag_keys=tag_keys,
        tag_values=tag_values,
        compute_tag_keys=compute_tag_keys,
        top_n=top_n
        )
       
       return [], updated_data.to_dict('records'), {'display': 'none'}, dash.no_update, dash.no_update, dash.no_update


    ##### Handle cell change
    if triggered_id == 'adhoc-usage-ag-grid' and cell_change:


        if changes is None:
            changes = []
        change_data = cell_change[0]['data']
        row_index = cell_change[0]['rowIndex']
        # Ensure the change data includes the row index
        change_data['rowIndex'] = row_index


        changes.append(change_data)

        row_data = mark_changed_rows(row_data, changes, row_id='rowIndex')

        return changes, row_data, {'display': 'block', 'color': 'yellow', 'font-weight': 'bold'}, dash.no_update, dash.no_update, dash.no_update


    ##### Handle Clear Button Press
    elif triggered_id == 'usage-adhoc-clear-btn' and clear_clicks:
        
        clear_loading_content = html.Button('Clear Adhoc Changes', id='usage-adhoc-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')

        return [], original_data, {'display': 'none'}, dash.no_update, clear_loading_content, dash.no_update
    

    ##### Handle Saving New Tag Key Values 
    ##### SAVE CHANGES

    # Handle saving changes
    if triggered_id == 'usage-adhoc-save-btn' and save_clicks and changes:

        # Combine changes by row index
        grouped_changes = []

        if changes:
            grouped_changes = group_changes_by_row(changes) ## from data_functions.utils import *


        connection = system_query_manager.get_engine().connect()

        save_loading_content = html.Button('Save Changes', id='usage-adhoc-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className='prettier-button')

        ### MERGE KEY IS cluster_id, tag_key_name
        ## We Basically dont need a separate UPDATE / INSERT because the cluster exists already. We simply MERGE UPSERT for each cluster/ ke/value
        if changes:
            try:
                # Process grouped changes for both updates and inserts
                for change in grouped_changes:
                    #print("Combined change data:", change)  # Debug statement
                    record_id = change.get('cluster_id')

                    if record_id:
                        # Update existing record
                        update_query = text("""
                            WITH updates AS (
                                SELECT 
                                :cluster_id AS compute_asset_id,
                                'ALL_PURPOSE' AS compute_asset_type,
                                NULL AS tag_policy_name,
                                :tag_key AS tag_key,
                                :tag_value AS tag_value,
                                now() AS update_timestamp,
                                false AS is_persisted_to_actual_asset  
                            )
                            MERGE INTO app_compute_tags t
                            USING updates u ON 
                                        u.compute_asset_id = t.compute_asset_id
                                        AND u.compute_asset_type = t.compute_asset_type
                                        AND u.tag_key = t.tag_key
                            WHEN MATCHED THEN 
                                UPDATE
                            SET compute_asset_id = u.compute_asset_id,
                                compute_asset_type = u.compute_asset_type,
                                tag_policy_name = u.tag_policy_name,
                                tag_key = u.tag_key,
                                tag_value = u.tag_value,
                                update_timestamp = u.update_timestamp
                            WHEN NOT MATCHED THEN INSERT (compute_asset_id, compute_asset_type, tag_policy_name, tag_key, tag_value, update_timestamp)
                                            VALUES (u.compute_asset_id, u.compute_asset_type, u.tag_policy_name, u.tag_key, u.tag_value, u.update_timestamp)
                        """)

                        connection.execute(update_query, parameters={
                            'cluster_id': record_id,
                            'tag_key': change.get('input_policy_key', None),
                            'tag_value': change.get('input_policy_value', None)
                        })

                        connection.commit()

            except Exception as e:
                print(f"Error during save with changes: {changes}")  # Debug error
                raise e
            finally:
                connection.close()

        ## No change, no Op
        else:
            print(f"NOOPPPP for some reason: {changes} \n {original_data}")
            pass
            
        
        return [], original_data, {'display': 'none'}, save_loading_content, dash.no_update, 'tag_save_triggered'
        

    # Default return to avoid callback errors
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update



#### JOBS AG Grid Callback
@app.callback(
    [Output('jobs-ag-grid-store', 'data'),
     Output('jobs-usage-ag-grid', 'rowData'),
     Output('usage-jobs-change-indicator', 'style'),
     Output('loading-save-jobs-usage', 'children'),
     Output('loading-clear-jobs-usage', 'children'),
     Output('save-jobs-trigger', 'children')],
    ## Also need to update all filters and visuals? Or let the button to do? with the "Refresh Button"
    Input('update-params-button', 'n_clicks'),
    State('start-date-picker', 'date'),
    State('end-date-picker', 'date'),
    State('tag-filter-dropdown', 'value'),
    State('product-category-dropdown', 'value'), ## Tuple
    State('tag-policy-dropdown', 'value'), ## Tuple
    State('tag-policy-key-dropdown', 'value'), ## Tuple
    State('tag-policy-value-dropdown', 'value'), ## Tuple
    State('compute-tag-filter-dropdown', 'value'), ## Tuple
    ### Actual AG Grid State
    Input('usage-jobs-save-btn', 'n_clicks'), 
     Input('usage-jobs-clear-btn', 'n_clicks'),
     Input('jobs-usage-ag-grid', 'cellValueChanged'),
     Input('top-n-jobs', 'value'), 
    State('jobs-usage-ag-grid', 'rowData'),
     State('jobs-ag-grid-store', 'data'),
     State('jobs-ag-grid-original-data', 'data'),
     State('jobs-usage-ag-grid', 'selectedRows')
)
def update_jobs_grid_data(n_clicks, start_date, end_date, tag_filter, product_category, tag_policies, tag_keys, tag_values, compute_tag_keys,
                           save_clicks, clear_clicks, cell_change, top_n,
                           row_data, changes, original_data, selected_rows):

    
    ### Synchronously figure out what action is happening and run the approprite logic. 
    ## This single-callback method ensure that no strange operation order can happen. Only one can happen at once. 
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]


    ##### Handle Changes to top N filter
    if (triggered_id == 'top-n-jobs' and top_n) or (triggered_id == 'update-params-button' and n_clicks > 0):
       
       updated_data = tag_advisor_manager.get_jobs_clusters_grid_data(
        start_date=start_date,
        end_date=end_date,
        tag_filter=tag_filter,
        product_category = product_category,
        tag_policies=tag_policies,
        tag_keys=tag_keys,
        tag_values=tag_values,
        compute_tag_keys=compute_tag_keys,
        top_n=top_n
        )
       
       return [], updated_data.to_dict('records'), {'display': 'none'}, dash.no_update, dash.no_update, dash.no_update


    ##### Handle cell change
    if triggered_id == 'jobs-usage-ag-grid' and cell_change:

        if changes is None:
            changes = []
        change_data = cell_change[0]['data']
        row_index = cell_change[0]['rowIndex']
        # Ensure the change data includes the row index
        change_data['rowIndex'] = row_index
        changes.append(change_data)
        row_data = mark_changed_rows(row_data, changes, row_id='rowIndex')

        return changes, row_data, {'display': 'block', 'color': 'yellow', 'font-weight': 'bold'}, dash.no_update, dash.no_update, dash.no_update


    ##### Handle Clear Button Press
    elif triggered_id == 'usage-jobs-clear-btn' and clear_clicks:
        
        clear_loading_content = html.Button('Clear Jobs Changes', id='usage-jobs-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')

        return [], original_data, {'display': 'none'}, dash.no_update, clear_loading_content, dash.no_update
    

    ##### Handle Saving New Tag Key Values 
    ##### SAVE CHANGES

    # Handle saving changes
    if triggered_id == 'usage-jobs-save-btn' and save_clicks:

        # Combine changes by row index
        grouped_changes = []

        if changes:
            grouped_changes = group_changes_by_row(changes) ## from data_functions.utils import *


        connection = system_query_manager.get_engine().connect()

        save_loading_content = html.Button('Save Changes', id='usage-jobs-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className='prettier-button')

        ### MERGE KEY IS job_id, tag_key_name
        ## We Basically dont need a separate UPDATE / INSERT because the cluster exists already. We simply MERGE UPSERT for each cluster/ ke/value
        if changes:
            try:
                # Process grouped changes for both updates and inserts
                for change in grouped_changes:
                    #print("Combined change data:", change)  # Debug statement
                    record_id = change.get('job_id')

                    if record_id:
                        # Update existing record
                        update_query = text("""
                            WITH updates AS (
                                SELECT 
                                :job_id AS compute_asset_id,
                                'JOBS' AS compute_asset_type,
                                NULL AS tag_policy_name,
                                :tag_key AS tag_key,
                                :tag_value AS tag_value,
                                now() AS update_timestamp,
                                false AS is_persisted_to_actual_asset  
                            )
                            MERGE INTO app_compute_tags t
                            USING updates u ON 
                                        u.compute_asset_id = t.compute_asset_id
                                        AND u.compute_asset_type = t.compute_asset_type
                                        AND u.tag_key = t.tag_key
                            WHEN MATCHED THEN 
                                UPDATE
                            SET compute_asset_id = u.compute_asset_id,
                                compute_asset_type = u.compute_asset_type,
                                tag_policy_name = u.tag_policy_name,
                                tag_key = u.tag_key,
                                tag_value = u.tag_value,
                                update_timestamp = u.update_timestamp
                            WHEN NOT MATCHED THEN INSERT (compute_asset_id, compute_asset_type, tag_policy_name, tag_key, tag_value, update_timestamp)
                                            VALUES (u.compute_asset_id, u.compute_asset_type, u.tag_policy_name, u.tag_key, u.tag_value, u.update_timestamp)
                        """)

                        connection.execute(update_query, parameters={
                            'job_id': record_id,
                            'tag_key': change.get('input_policy_key', None),
                            'tag_value': change.get('input_policy_value', None)
                        })

                        connection.commit()

            except Exception as e:
                print(f"Error during save with changes: {changes}")  # Debug error
                raise e
            finally:
                connection.close()

        ## No change, no Op
        else:
            pass
            
        
        return [], original_data, {'display': 'none'}, save_loading_content, dash.no_update, 'tag_save_triggered'
        

    # Default return to avoid callback errors
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update





#### SQL AG Grid Callback
@app.callback(
    [Output('sql-ag-grid-store', 'data'),
     Output('sql-usage-ag-grid', 'rowData'),
     Output('usage-sql-change-indicator', 'style'),
     Output('loading-save-sql-usage', 'children'),
     Output('loading-clear-sql-usage', 'children'),
     Output('save-sql-trigger', 'children')],
    ## Also need to update all filters and visuals? Or let the button to do? with the "Refresh Button"
    Input('update-params-button', 'n_clicks'),
    State('start-date-picker', 'date'),
    State('end-date-picker', 'date'),
    State('tag-filter-dropdown', 'value'),
    State('product-category-dropdown', 'value'), ## Tuple
    State('tag-policy-dropdown', 'value'), ## Tuple
    State('tag-policy-key-dropdown', 'value'), ## Tuple
    State('tag-policy-value-dropdown', 'value'), ## Tuple
    State('compute-tag-filter-dropdown', 'value'), ## Tuple
    ### Actual AG Grid State
    Input('usage-sql-save-btn', 'n_clicks'), 
     Input('usage-sql-clear-btn', 'n_clicks'),
     Input('sql-usage-ag-grid', 'cellValueChanged'),
     Input('top-n-sql', 'value'), 
    [State('sql-usage-ag-grid', 'rowData'),
     State('sql-ag-grid-store', 'data'),
     State('sql-ag-grid-original-data', 'data'),
     State('sql-usage-ag-grid', 'selectedRows')]
)
def update_sql_grid_data(n_clicks, start_date, end_date, tag_filter, product_category, tag_policies, tag_keys, tag_values, compute_tag_keys,
                           save_clicks, clear_clicks, cell_change, top_n,
                           row_data, changes, original_data, selected_rows):

    
    ### Synchronously figure out what action is happening and run the approprite logic. 
    ## This single-callback method ensure that no strange operation order can happen. Only one can happen at once. 
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]


    ##### Handle Changes to top N filter
    if (triggered_id == 'top-n-adhoc' and top_n) or (triggered_id == 'update-params-button' and n_clicks > 0):
       
       updated_data = tag_advisor_manager.get_sql_clusters_grid_data(
        start_date=start_date,
        end_date=end_date,
        product_category = product_category,
        tag_filter = tag_filter,
        tag_policies=tag_policies,
        tag_keys=tag_keys,
        tag_values=tag_values,
        compute_tag_keys=compute_tag_keys,
        top_n=top_n
        )
       
       return [], updated_data.to_dict('records'), {'display': 'none'}, dash.no_update, dash.no_update, dash.no_update


    ##### Handle cell change
    if triggered_id == 'sql-usage-ag-grid' and cell_change:

        if changes is None:
            changes = []
        change_data = cell_change[0]['data']
        row_index = cell_change[0]['rowIndex']
        # Ensure the change data includes the row index
        change_data['rowIndex'] = row_index
        changes.append(change_data)
        row_data = mark_changed_rows(row_data, changes, row_id='rowIndex')

        return changes, row_data, {'display': 'block', 'color': 'yellow', 'font-weight': 'bold'}, dash.no_update, dash.no_update, dash.no_update


    ##### Handle Clear Button Press
    elif triggered_id == 'usage-sql-clear-btn' and clear_clicks:
        
        clear_loading_content = html.Button('Clear SQL Changes', id='usage-sql-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')

        return [], original_data, {'display': 'none'}, dash.no_update, clear_loading_content, dash.no_update
    

    ##### Handle Saving New Tag Key Values 
    ##### SAVE CHANGES

    # Handle saving changes
    if triggered_id == 'usage-sql-save-btn' and save_clicks:

        # Combine changes by row index
        grouped_changes = []

        if changes:
            grouped_changes = group_changes_by_row(changes) ## from data_functions.utils import *


        connection = system_query_manager.get_engine().connect()

        save_loading_content = html.Button('Save Changes', id='usage-sql-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className='prettier-button')

        ### MERGE KEY IS job_id, tag_key_name
        ## We Basically dont need a separate UPDATE / INSERT because the cluster exists already. We simply MERGE UPSERT for each cluster/ ke/value
        if changes:
            try:
                # Process grouped changes for both updates and inserts
                for change in grouped_changes:
                    #print("Combined change data:", change)  # Debug statement
                    record_id = change.get('warehouse_id')

                    if record_id:
                        # Update existing record
                        update_query = text("""
                            WITH updates AS (
                                SELECT 
                                :warehouse_id AS compute_asset_id,
                                'SQL' AS compute_asset_type,
                                NULL AS tag_policy_name,
                                :tag_key AS tag_key,
                                :tag_value AS tag_value,
                                now() AS update_timestamp,
                                false AS is_persisted_to_actual_asset  
                            )
                            MERGE INTO app_compute_tags t
                            USING updates u ON
                                        u.compute_asset_id = t.compute_asset_id
                                        AND u.compute_asset_type = t.compute_asset_type
                                        AND u.tag_key = t.tag_key
                            WHEN MATCHED THEN 
                                UPDATE
                            SET compute_asset_id = u.compute_asset_id,
                                compute_asset_type = u.compute_asset_type,
                                tag_policy_name = u.tag_policy_name,
                                tag_key = u.tag_key,
                                tag_value = u.tag_value,
                                update_timestamp = u.update_timestamp
                            WHEN NOT MATCHED THEN INSERT (compute_asset_id, compute_asset_type, tag_policy_name, tag_key, tag_value, update_timestamp)
                                            VALUES (u.compute_asset_id, u.compute_asset_type, u.tag_policy_name, u.tag_key, u.tag_value, u.update_timestamp)
                        """)

                        connection.execute(update_query, parameters={
                            'warehouse_id': record_id,
                            'tag_key': change.get('input_policy_key', None),
                            'tag_value': change.get('input_policy_value', None)
                        })

                        connection.commit()

            except Exception as e:
                print(f"Error during save with changes: {changes}")  # Debug error
                raise e
            finally:
                connection.close()

        ## No change, no Op
        else:
            pass
            
        
        return [], original_data, {'display': 'none'}, save_loading_content, dash.no_update, 'tag_save_triggered'
        

    # Default return to avoid callback errors
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update



@app.callback(
    [Output('cluster-tag-changes-store', 'data'),
     Output('cluster-tag-rowData-store', 'data'),
     Output('compute-tag-change-indicator', 'style'),
     Output('loading-save-compute-tags', 'children'),
     Output('loading-clear-compute-tags', 'children'),
     Output('save-tags-trigger', 'children'), ## For when we just need to reload data locally
     Output('reload-tags-trigger', 'children')],
    ## Also need to update all filters and visuals? Or let the button to do? with the "Refresh Button"
   [Input('tag-compute-tags-save-btn', 'n_clicks'), 
     Input('tag-compute-tags-clear-btn', 'n_clicks'),
     Input('add-compute-tag-row-btn', 'n_clicks'),
     Input('remove-compute-tags-row-btn', 'n_clicks'),
     Input('tag-compute-ag-grid', 'cellValueChanged')], 
    [State('cluster-tag-changes-store', 'data'),
     State('tag-compute-ag-grid', 'rowData'),
     State('cluster-tag-rowData-store', 'data'),
     State('tag-compute-ag-grid', 'selectedRows')]
)
def update_tag_compute_grid_data(save_clicks, clear_clicks, add_row_clicks, remove_row_clicks, cell_change,
                           changes, row_data, original_data, selected_rows):

    
    ### Synchronously figure out what action is happening and run the approprite logic. 
    ## This single-callback method ensure that no strange operation order can happen. Only one can happen at once. 
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]


    #### TO DO: Need to Handle Deletes
    ##### CREATE - Add New row in GRID
    if triggered_id == 'add-compute-tag-row-btn' and add_row_clicks > 0:

        new_row = {
            'tag_id': None,  # Will be generated by the database
            'compute_asset_id': '',
            'compute_asset_type': '',
            'tag_policy_name': '',
            'tag_key': '',
            'tag_value': '',
            'is_persisted_to_actual_asset': None
        }
        row_data.append(new_row)

        return dash.no_update, row_data, dash.no_update, dash.no_update, dash.no_update, 'Added Tag Row To Grid', dash.no_update


    ##### DELETES 
    #### DELETE Handle removing selected rows
    
    if triggered_id == 'remove-compute-tags-row-btn' and remove_row_clicks > 0:
        ## Only attempt to delete from

        ids_to_remove = [row['tag_id'] for row in selected_rows if row['tag_id'] is not None]
        updated_row_data = [row for row in row_data if row['tag_id'] not in ids_to_remove]
        
        if ids_to_remove:
            connection = system_query_manager.get_engine().connect()
            try:
                delete_query = text("""
                    DELETE FROM app_compute_tags
                    WHERE tag_id IN :ids
                """).bindparams(bindparam('ids', expanding=True))
                connection.execute(delete_query, parameters= {'ids':ids_to_remove})
                connection.commit()

            except Exception as e:
                print(f"Error during deletion: {e}")
                raise e
            finally:
                connection.close()

        if changes is not None:
            updated_changes = [change for change in changes if change['tag_id'] not in ids_to_remove]
        else:
            updated_changes = []


        return updated_changes, updated_row_data, dash.no_update, dash.no_update, dash.no_update, 'Deleted Row Locally and in DB', dash.no_update



    ##### Handle cell change
    if triggered_id == 'tag-compute-ag-grid' and cell_change:

        if changes is None:
            changes = []
        change_data = cell_change[0]['data']
        row_index = cell_change[0]['rowIndex']
        # Ensure the change data includes the row index
        change_data['rowIndex'] = row_index
        changes.append(change_data)
        row_data = mark_changed_rows(row_data, changes, row_id='rowIndex')

        return changes, row_data, {'display': 'block', 'color': 'yellow', 'font-weight': 'bold'}, dash.no_update, dash.no_update, 'Edited Row Locally', dash.no_update


    ##### Handle Clear Button Press
    elif triggered_id == 'ag-compute-tags-clear-btn' and clear_clicks:
        
        clear_loading_content = html.Button('Clear Tag Changes', id='tag-compute-tags-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')

        return [], original_data, {'display': 'none'}, dash.no_update, clear_loading_content, dash.no_update, 'Cleared All Changes and Reload'
    

    ##### Handle Saving New Tag Key Values 
    ##### SAVE CHANGES

    # Handle saving changes
    if triggered_id == 'tag-compute-tags-save-btn' and save_clicks:

        # Combine changes by row index
        grouped_changes = []

        if changes:
            grouped_changes = group_changes_by_row(changes) ## from data_functions.utils import *


        connection = system_query_manager.get_engine().connect()

        save_loading_content = html.Button('Save Tag Changes', id='tag-compute-tags-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')

        ### MERGE KEY IS job_id, tag_key_name
        ## We Basically dont need a separate UPDATE / INSERT because the cluster exists already. We simply MERGE UPSERT for each cluster/ ke/value
        if changes:
            try:
                # Process grouped changes for both updates and inserts
                for change in grouped_changes:
                    #print("Combined change data:", change)  # Debug statement
                    record_id = change.get('tag_id')

                    if record_id:
                        # Update existing record
                        update_query = text("""
                            UPDATE app_compute_tags t
                            SET 
                            tag_policy_name = :tag_policy_name,
                            tag_key = :tag_key,
                            tag_value = :tag_value
                            WHERE tag_id = :tag_id
                                    AND compute_asset_id = :compute_asset_id
                                    AND compute_asset_type = :compute_asset_type
                               
                        """)

                        connection.execute(update_query, parameters={
                            'tag_id': record_id,
                            'compute_asset_id': change.get('compute_asset_id', None),
                            'compute_asset_type': change.get('compute_asset_type', None),
                            'tag_policy_name': change.get('tag_policy_name', None),
                            'tag_key': change.get('tag_key', None),
                            'tag_value': change.get('tag_value', None)
                        })

                        connection.commit()

            except Exception as e:
                print(f"Error during save with tag changes: {changes}")  # Debug error
                raise e
            finally:
                connection.close()

        ## No change, no Op
        else:
            pass

        updated_data_after_save = tag_advisor_manager.get_compute_tagged_grid_data().to_dict('records')
            
        return [], updated_data_after_save, {'display': 'none'}, save_loading_content, dash.no_update, dash.no_update, 'tag_save_triggered'
        

    # Default return to avoid callback errors
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update



@app.callback(
    Output('tag-compute-ag-grid', 'rowData'),
    [Input('save-adhoc-trigger', 'children'),
     Input('save-jobs-trigger', 'children'),
     Input('save-sql-trigger', 'children'),
     Input('save-tags-trigger', 'children'),
     Input('reload-tags-trigger', 'children')],
     [State('cluster-tag-rowData-store', 'data')]
)
def update_tag_compute_ag_grid(adhoc, jobs, sql, tags, reloadTrigger, rowData):

    if adhoc or jobs or sql:
        # Logic to fetch or compute new rowData for 'tag-compute-ag-grid'
        updated_data = tag_advisor_manager.get_compute_tagged_grid_data().to_dict('records')
        return updated_data
    
    if tags:
        current_state = rowData

        return current_state
    
    if reloadTrigger:
        return tag_advisor_manager.get_compute_tagged_grid_data().to_dict('records')

    
    return dash.no_update


##### LLM Alert Manager #####

@app.callback(
    [Output('chat-history', 'data'),
     Output('chat-output-window', 'children'),
     Output('chat-input-box', 'value'),
     Output('in-progress-alert', 'data')],
    [Input('chat-submit-btn', 'n_clicks'),
     Input('clear-context-btn', 'n_clicks')],
    [State('chat-input-box', 'value'),
     State('chat-history', 'data')],
    prevent_initial_call=True
)
def submit_chat(chat_submit_n_clicks, clear_context_n_clicks, input_value, history):

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'clear-context-btn':
        # Clear the chat history
        return {'messages': []}, "Chat history cleared.", "", {'data': {'alert': {}}}

    if button_id == 'chat-submit-btn' and input_value:
        new_history = history['messages'][-4:]  # keep only the last 2 messages for rolling basis
        new_input = """**ME**:   """+ input_value
        new_history.append(new_input)
        response_message = ""

        # Formulate the input for AI Query
        ### each new prompt/response combo 
        chat_updated_input = '\n\n'.join(new_history)
        query = text("SELECT result FROM generate_alert_info_from_prompt(:input_prompt)")

        try:
            # Assuming system_query_manager.get_engine() is predefined and correct
            engine = system_query_manager.get_engine()
            with engine.connect() as conn:
                result = conn.execute(query, parameters={'input_prompt': chat_updated_input})
                row = result.fetchone()
                new_output = row[0]
                if row:
                    response_message = """**DBRX**:    """ + new_output
                    
                else:
                    response_message = """**DBRX**:    """ + "No response generated."
                new_history.append(response_message)

        except Exception as e:
            response_message = """**DBRX**:   """ + f"ERROR getting alert data: {str(e)}"
            new_history.append(response_message)

        #print(f"CONTEXT CHAIN: \n{new_history}")
        #print(f"CURRENT OUTPUT: \n {new_output}")

        parsed_updated_output = parse_query_result_json_from_string(new_output)
        print(parsed_updated_output)
        #print(parsed_updated_output)
        return {'messages': new_history}, '\n\n'.join(new_history), "", {'alert': parsed_updated_output}

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update



### Callback to fill the live alerts values
@app.callback(
    Output('input-alert-name', 'value'),
    Output('input-query', 'value'),
    Output('input-schedule', 'value'),
    Output('input-recipients', 'value'),
    Output('input-context-sql', 'value'),
    Input('in-progress-alert', 'data'), 
    prevent_initial_call=True
)
def update_inputs_from_store(data):

    alert_data = data.get('alert', dict())
    
    alert_name = alert_data.get('ALERT_NAME', '')
    query = alert_data.get('QUERY', '')
    schedule = alert_data.get('SCHEDULE', '')
    recipients = ', '.join(alert_data.get('RECIPIENTS', []))
    context_sql = ', '.join(alert_data.get('CONTEXT_SQL', []))
    
    return alert_name, query, schedule, recipients, context_sql



# ALERTS AG Grid Callback to update the Alert grid with data from the database
@app.callback(
    Output('alerts-grid', 'rowData'),
    Output('save-pending-alert-loading', 'children'),
    Output('loading-remove-alerts', 'children'),
    Output('loading-create-jobs-alerts', 'children'),
    Output('loading-refresh-alerts', 'children'),
    ## the in progress alert store should also be cleared, so do we combine callbacks?
    Input('save-pending-alerts-btn', 'n_clicks'),
    Input('create-alert-jobs-btn', 'n_clicks'),
    Input('remove-alerts-btn', 'n_clicks'),
    Input('refresh-alerts-grid-btn', 'n_clicks'),
    State('in-progress-alert', 'data'),
    State('alerts-grid', 'rowData'),
    State('alerts-grid', 'selectedRows') ## Might not need a store because we are just going to delete selected rows, no adding / editing
)
def update_alerts_ag_grid(save_clicks, create_jobs_clicks, remove_alerts_clicks, refresh_grid_clicks, in_progress_data, row_data, selected_rows):

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    action_id = ctx.triggered[0]['prop_id'].split('.')[0]


    if action_id == 'save-pending-alerts-btn' and save_clicks > 0:

        ###### Handle saves from the LLM
        alert_data = in_progress_data.get('alert', dict())
        
        alert_name = alert_data.get('ALERT_NAME', None)
        query = alert_data.get('QUERY', None)
        schedule = alert_data.get('SCHEDULE', None)
        recipients = ', '.join(alert_data.get('RECIPIENTS', []))
        context_sql = ', '.join(alert_data.get('CONTEXT_SQL', []))

        #### Save to SQL and return results

        ### Do input validation here
        if (len(alert_name) > 0) and (len(query)> 0): ## i.e. has pending changes
            ## Instead of having changes from AG grid, we only have single alert changes at a given time. static parsed values


            connection = system_query_manager.get_engine().connect()

            save_pending_alerts_btn = html.Button('Save Alert', id='save-pending-alerts-btn', n_clicks=0,
                                    className = 'prettier-button')

        ### MERGE KEY IS query_text, query_name
            try:

                        # Do NOT update a record with the same name. Must manually delete it
                        upsert_query = text("""
                            WITH updates AS (
                                SELECT 
                                :alert_name AS alert_name,
                                :alert_query AS alert_query,
                                :alert_schedule AS alert_schedule,
                                :alert_recipients AS alert_recipients
                            )
                            MERGE INTO alerts_settings t
                            USING updates u ON 
                                        u.alert_name = t.alert_name
                                        AND u.alert_query = t.alert_query
                            WHEN NOT MATCHED THEN INSERT (alert_name, alert_query, alert_schedule, alert_recipients)
                                            VALUES (u.alert_name, u.alert_query, u.alert_schedule, u.alert_recipients)
                        """)

                        connection.execute(upsert_query, parameters={
                            'alert_name': alert_name,
                            'alert_query': query,
                            'alert_schedule': schedule,
                            'alert_recipients': recipients
                        })

                        connection.commit()

            except Exception as e:
                print(f"Error during save with alert changes: {alert_data}")  # Debug error
                raise e
            finally:
                connection.close()

        ## No change, no Op
        else:
            pass

        updated_data_after_save = alerts_manager.get_alerts_ag_grid_data().to_dict('records')
            
        return updated_data_after_save, save_pending_alerts_btn, dash.no_update, dash.no_update, dash.no_update
        

    ###### Handle Deletes

    if action_id == 'remove-alerts-btn' and remove_alerts_clicks > 0:
        ## Only attempt to delete from
        updated_delete_button = html.Button('Delete Rows', id='remove-alerts-btn', n_clicks=0,
                                    className = 'prettier-button', style={'margin-bottom': '10px'})
        ## Signature of rows to remove (name + query)
        ids_to_remove = [row['id'] for row in selected_rows if row['alert_query'] is not None]

        ## Get alerts to delete the jobs as well
        rows_to_remove = [row for row in row_data if row['id'] in ids_to_remove]

        updated_row_data = [row for row in row_data if row['id'] not in ids_to_remove]
        
        if ids_to_remove:

            connection = system_query_manager.get_engine().connect()
            try:
                delete_query = text("""
                    DELETE FROM alerts_settings
                    WHERE id IN :ids
                """).bindparams(bindparam('ids', expanding=True))
                
                connection.execute(delete_query, parameters={
                            'ids': ids_to_remove
                        })
                
                connection.commit()

            except Exception as e:
                print(f"Error during deletion: {e}")
                raise e
            finally:
                connection.close()

        if rows_to_remove:
            ## Do do not go back to database just to delete something and re-trieve data we already have

            for alert in rows_to_remove:

                ### Not all alerts will have these, so the function looks for emptiness and deletes them indepdendently

                job_id = alert.get('job_id', None)
                query_id = alert.get('query_id', None)
                alert_id = alert.get('alert_id', None)

                ## Delete the associated queries, alerts, jobs to ensure things dont get cluttered
                ## Function handles empty values
                delete_alert_and_job(dbx_client=dbx_client, query_id=query_id, alert_id=alert_id, job_id=job_id)
                ######

            updated_saved_rows = alerts_manager.get_alerts_ag_grid_data().to_dict('records')

        else:
            ## If no data to remove, just return same data
            updated_saved_rows = row_data

        return updated_saved_rows, dash.no_update, updated_delete_button, dash.no_update, dash.no_update


    ## Manual refresh button        
    if action_id == 'refresh-alerts-grid-btn' and refresh_grid_clicks > 0:

        refresh_alert_button = html.Button('Refresh', id='refresh-alerts-grid', n_clicks=0,
                                    className = 'prettier-button', style={'margin-bottom': '10px'})
        updated_grid_data_refresh = alerts_manager.get_alerts_ag_grid_data().to_dict('records')

        return updated_grid_data_refresh, dash.no_update, dash.no_update, dash.no_update, refresh_alert_button



    ###### Handle job/query creation and metadata saving
    if action_id == 'create-alert-jobs-btn' and create_jobs_clicks > 0 and selected_rows:

        connection = system_query_manager.get_engine().connect()

        save_loading_jobs = html.Button('Create Alert Jobs', id='create-alert-jobs-btn', n_clicks=0,
                                    className = 'prettier-button', style={'margin-bottom': '10px'})
        ##### use Databricks SDK to create the query, alert, and job from the row text of selected rows
        
        #### Step 1 - for each alert, create job and return results. Save to data frame
        for new_row in selected_rows:

            new_record_id = new_row.get('id')
            new_alert_query = new_row.get('alert_query')
            new_alert_name = new_row.get('alert_name')
            new_alert_schedule = new_row.get('alert_schedule')
            new_alert_recipients_str = new_row.get('alert_recipients')
            new_alert_recipients = []

            if new_alert_recipients_str is not None:
                new_alert_recipients.append(new_alert_recipients_str.split(","))

            ## New job id data
            job_dict = create_alert_and_job(dbx_client = dbx_client, 
                         warehouse_id = warehouse_id, 
                         alert_name = new_alert_name, 
                         alert_query= new_alert_query,
                         alert_schedule = new_alert_schedule, 
                         subscribers = new_alert_recipients
                         )
            
            print(f"CREATED JOBS FOR ALERT: {job_dict}")
            
            new_job_id = job_dict.get('job_id')
            new_query_id = job_dict.get('query_id')
            new_alert_id = job_dict.get('alert_id')

            #### Step 2 - Update grid / SQL in result db

            try:

                print(f"UPDATING ALERT ID: {new_record_id}")

                if new_alert_name and new_alert_query:
                    # Update existing record
                    update_query = text("""
                        UPDATE alerts_settings t
                        SET 
                        job_id = :job_id,
                        alert_id = :alert_id,
                        query_id = :query_id
                        WHERE id = :new_alert_id
                            
                    """)

                    connection.execute(update_query, parameters={
                        'new_alert_id': new_record_id,
                        'job_id': new_job_id,
                        'alert_id': new_alert_id,
                        'query_id': new_query_id
                    })

                    connection.commit()

            except Exception as e:
                print(f"Error during save with alert job creation: {new_row} \n {str(e)}")  # Debug error
                raise e
            finally:
                connection.close()


        #### Step 3 - return results - after ALL rows have been updated, not each row
        updated_saved_rows = alerts_manager.get_alerts_ag_grid_data()

        return updated_saved_rows.to_dict('records'), dash.no_update, dash.no_update, save_loading_jobs, dash.no_update


    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


##### END APP #####

if __name__ == '__main__':
    app.run_server(debug=True)
