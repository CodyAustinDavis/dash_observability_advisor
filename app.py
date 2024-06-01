
"""
Author: Cody Austin Davis
Date: 5/25/2024
Description: Dash Apps for Tagging and Data Compute Observability

TO DO: 
1. Add databricsk SDK for authentication and iteraction with db
2. Create SQL Alchemy backend and engine - DONE
3. Create all visuals
4. Create AG Grid for tags
5. Create job button for syncing tags with app (tags jobs that arent tagged yet)
6. Create settings page that lets a user input the workspace name and select the warehouse to execute against. (OAuth)
7. Create LLM to generate visuals for system tables upon request
8. Create Materialized View

"""


import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from databricks import sql
import os
from dotenv import load_dotenv
from page_functions import *
from data_functions import *
from chart_functions import ChartFormats
import plotly.express as px


## Log SQL Alchemy Commands
#logging.basicConfig(level=logging.ERROR)
#logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)

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


## Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


## Load Base Query Template Parts
TAG_QUERY_1 = read_sql_file("./config/tag_query_1.sql") ## Where you add selected tags
TAG_QUERY_2 = read_sql_file("./config/tag_query_2.sql")
TAG_QUERY_3 = read_sql_file("./config/tag_query_3.sql") ## Where you filter the rest of the paramters
TAG_QUERY_4 = read_sql_file("./config/tag_query_4.sql") ## Where you filter the rest of the paramters


##### Load Dynamic SQL Files
def build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, 
                                start_date, end_date, 
                                product_category = None, 
                                tag_policies = None, tag_keys = None, tag_values = None, 
                                final_agg_query=None):

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

    
    ## Final Select Statement -- this is after all the standard server-side filtering

    if final_agg_query:
        FINAL_QUERY = FINAL_QUERY + ")\n" + final_agg_query
    
    else:
        FINAL_QUERY = FINAL_QUERY + TAG_QUERY_4

    return FINAL_QUERY



# Define the layout of the main app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Nav(
                [
                html.Div([  # Container for the logo and possibly other header elements
                    html.Img(src='/assets/app_logo.png', style={'width': '100%', 'height': 'auto'}),
                    ]),
                dbc.NavLink("Tagging Advisor", href="/tag-advisor", id="tab-1-link"),
                dbc.NavLink("Alert Manager", href="/alert-manager", id="tab-2-link"),
                dbc.NavLink("Settings", href="/settings", id="tab-3-link"),
                dbc.NavLink("Model Settings", href="/model-settings", id="tab-4-link")
                ],
                vertical=True,
                pills=True,
                className="sidebar"  # Add the class name here
            )
        ], width={"size": 2, "offset": 0}),  # Width of the sidebar
        dbc.Col([
            dcc.Location(id='url', refresh=False),
            html.Div(id='tab-content', className="main-content")  # Add the class name here
        ], width=10)  # Width of the main content
    ])
], fluid=True)


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
    return pathname == "/tag-advisor", pathname == "/alert-manager", pathname == "/settings", pathname == "/model-settings"


# Callback to update the content based on tab selection
@app.callback(
    Output('tab-content', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):

    ## TO DO: Add authentication screen
    ## Can one of these be called from an LLM to create visuals???

    if pathname == "/tag-advisor":
        return render_tagging_advisor_page()
    elif pathname == "/alert-manager":
        return render_alert_manager_page()
    elif pathname == "/settings":
        return render_settings_page()
    elif pathname == "/model-settings":
        return render_model_settings_page()
    else:
        return render_tagging_advisor_page()



##### Update Usage Chart


@app.callback(
    Output('usage-by-match-chart', 'figure'),
    Input('update-params-button', 'n_clicks'),
    State('start-date-picker', 'date'),
    State('end-date-picker', 'date'),
    State('product-category-dropdown', 'value'), ## Tuple
    State('tag-policy-dropdown', 'value'), ## Tuple
    State('tag-key-dropdown', 'value'), ## Tuple
    State('tag-value-dropdown', 'value'), ## Tuple
)
def update_usage_by_match(n_clicks, start_date, end_date, product_category, tag_policies, tag_keys, tag_values):

    print(f"Clicked: {n_clicks}, Start Date: {start_date}, End Date: {end_date}")
    print(f"Tag Policies: {tag_policies}, Tag Keys: {tag_keys}, Tag Values: {tag_values}")

    # Define the color mapping for compliance
    color_map = {
        'In Policy': '#097969',
        'Not Matched To Tag Policy': '#002147'
    }

    AGG_QUERY = """
                SELECT usage_date AS Usage_Date, 
                SUM(usage_quantity) AS Usage_Quantity,
                IsTaggingMatch AS Tag_Match
                FROM filtered_result
                GROUP BY usage_date, IsTaggingMatch
                """

    ## If no clicks, use default paramters - just date range defaults
    if n_clicks == 0:

        query = build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, start_date=start_date, end_date=end_date, product_category= product_category, final_agg_query=AGG_QUERY)

                # Convert the result to a DataFrame
        df = system_query_manager.execute_query_to_df(query)

        # Create a Plotly Express line chart
        fig = px.bar(df, x='Usage_Date', y='Usage_Quantity', 
                     title='Daily Usage By Tag Policy Match',
                     color='Tag_Match',
                     labels={'Usage_Quantity': 'Usage Quantity', 'Usage_Date': 'Usage Date'},
                     barmode='stack',
                     color_discrete_map=color_map)
        fig.update_layout(ChartFormats.common_chart_layout())
        fig.update_layout(
                legend=dict(
                    x=0,
                    y=-0.3,  # You may need to adjust this depending on your specific plot configuration
                    orientation="h"  # Horizontal legend
                )
            )

        return fig

    elif n_clicks > 0:

        # Convert the result to a DataFrame
        query = build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, start_date=start_date, end_date=end_date, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=AGG_QUERY)

                # Convert the result to a DataFrame
        df = system_query_manager.execute_query_to_df(query)

        # Create a Plotly Express line chart
        fig = px.bar(df, x='Usage_Date', y='Usage_Quantity', 
                     title='Daily Usage By Tag Policy Match',
                     color='Tag_Match',
                     labels={'Usage_Quantity': 'Usage Quantity', 'Usage_Date': 'Usage Date'},
                     barmode='stack',
                     color_discrete_map=color_map)
        
        fig.update_layout(ChartFormats.common_chart_layout())
        fig.update_layout(
                legend=dict(
                    x=0,
                    y=-0.3,  # You may need to adjust this depending on your specific plot configuration
                    orientation="h"  # Horizontal legend
                )
            )

        return fig
    
    else: 
        return None



if __name__ == '__main__':

    app.run_server(debug=True)
