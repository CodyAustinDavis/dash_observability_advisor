
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


BUGS: 
1. Make filters update based on other selected filter values

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
import plotly.graph_objects as go
import numpy as np
import pandas as pd


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


## Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


## Load Base Query Template Parts
TAG_QUERY_1 = read_sql_file("./config/tag_query_1.sql") ## Where you add selected tags
TAG_QUERY_2 = read_sql_file("./config/tag_query_2.sql")
TAG_QUERY_3 = read_sql_file("./config/tag_query_3.sql") ## Where you filter the rest of the paramters
TAG_QUERY_4 = read_sql_file("./config/tag_query_4.sql") ## Where you filter the rest of the paramters


##### Load Dynamic SQL Files
def build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, 
                                start_date, end_date, tag_filter,
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
    Output('matched-usage-ind', 'figure'),
    Output('unmatched-usage-ind', 'figure'),
    Output('usage-by-tag-value-chart', 'figure'),
    Output('usage-heatmap', 'figure'),
    Input('update-params-button', 'n_clicks'),
    State('tag-filter-dropdown', 'value'),
    State('start-date-picker', 'date'),
    State('end-date-picker', 'date'),
    State('product-category-dropdown', 'value'), ## Tuple
    State('tag-policy-dropdown', 'value'), ## Tuple
    State('tag-key-dropdown', 'value'), ## Tuple
    State('tag-value-dropdown', 'value'), ## Tuple
)
def update_usage_by_match(n_clicks, tag_filter, start_date, end_date, product_category, tag_policies, tag_keys, tag_values):

    # Define the color mapping for compliance
    color_map = {
        'In Policy': '#097969',
        'Not Matched To Tag Policy': '#002147'
    }

    AGG_QUERY = """
                SELECT usage_date AS Usage_Date, 
                SUM(usage_quantity) AS Usage_Quantity,
                IsTaggingMatch AS `Tag Match`
                FROM filtered_result
                GROUP BY usage_date, IsTaggingMatch
                """

    MATCHED_IND_QUERY = """
                        SELECT 
                        SUM(CASE WHEN IsTaggingMatch = 'In Policy' THEN usage_quantity ELSE 0 END) AS `Matched Usage Quantity`,
                        SUM(CASE WHEN IsTaggingMatch = 'Not Matched To Tag Policy' THEN usage_quantity ELSE 0 END) AS `Not Matched Usage Quantity`
                        FROM filtered_result
                        """
    

    TAG_VALUES_QUERY = """
                 , exploded_tags AS 
                    (SELECT
                    explode(TagCombos) AS PolicyTagValue,
                    usage_quantity AS Usage_Quantity
                    FROM 
                    filtered_result
                    )

                    SELECT CASE WHEN len(PolicyTagValue) = 0 
                            THEN 'Not In Policy' ELSE PolicyTagValue END 
                            AS `Tag Value In Policy`,
                     SUM(Usage_Quantity) AS `Usage Quantity`
                    FROM exploded_tags
                    GROUP BY CASE WHEN len(PolicyTagValue) = 0 THEN 'Not In Policy' ELSE PolicyTagValue END 
                    ORDER BY `Usage Quantity` DESC
                """
    
    HEATMAP_QUERY = """ 
                    , exploded_tags AS (
                    SELECT
                        explode(TagCombos) AS PolicyTagValue,
                        usage_quantity AS Usage_Quantity,
                        billing_origin_product AS Product
                    FROM
                        filtered_result
                    )

                    SELECT 
                    PolicyTagValue AS Tag,
                    Product AS Product,
                    SUM(usage_quantity) AS `Usage Quantity`
                    FROM exploded_tags
                    GROUP BY  PolicyTagValue,
                    Product
                """
    

    # Convert the result to a DataFrame
    query = build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, tag_filter=tag_filter, start_date=start_date, end_date=end_date, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=AGG_QUERY)
    # Convert the result to a DataFrame
    df = system_query_manager.execute_query_to_df(query)

    # Create a Plotly Express line chart
    fig = px.bar(df, x='Usage_Date', y='Usage_Quantity', 
                    title='Daily Usage By Tag Policy Match',
                    color='Tag Match',
                    labels={'Usage_Quantity': 'Usage Quantity', 'Usage_Date': 'Usage Date'},
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
    ind_query = build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, tag_filter=tag_filter, start_date=start_date, end_date=end_date, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=MATCHED_IND_QUERY)
    ind_df = system_query_manager.execute_query_to_df(ind_query)
    matched_value = safe_round(ind_df['Matched Usage Quantity'][0], 0)
    not_matched_value = safe_round(ind_df['Not Matched Usage Quantity'][0], 0)


    matched_fig = go.Figure(go.Indicator(
                            mode="number+gauge",
                            value=matched_value,
                            title={"text": "Matched Usage", 'font': {'size': 18}},
                            number={'font': {'size': 24, 'color': "#097969"}, 'valueformat': ','},
                            gauge={'shape': "angular",
                            'axis': {'range': [0, matched_value + not_matched_value]},
                            'bar': {'color': "#097969"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#002147"},
                            domain = {'x': [0, 1], 'y': [0, 0.9]}  # Adjust domain to fit elements
                            ))
    
    matched_fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    #### Not Matched Usage Indicator
    unmatched_fig = go.Figure(go.Indicator(
                            mode="number+gauge",
                            value=not_matched_value,
                            title={"text": "Not Matched Usage", 'font': {'size': 18}},
                            number={'font': {'size': 24, 'color': '#8B0000'}, 'valueformat': ','},
                            gauge={'shape': "angular",
                            'axis': {'range': [0, matched_value + not_matched_value]},
                            'bar': {'color': '#8B0000'},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#002147"},
                            domain = {'x': [0, 1], 'y': [0, 0.9]}  # Adjust domain to fit elements
                            ))
    
    unmatched_fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=50, b=10)
    )


    #### Usage By Tag Value Bar Chart
    values_query = build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, 
                                               tag_filter=tag_filter, start_date=start_date, end_date=end_date, 
                                               product_category=product_category, 
                                               tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, 
                                               final_agg_query=TAG_VALUES_QUERY)
    
    values_df = system_query_manager.execute_query_to_df(values_query).sort_values(by='Usage Quantity', ascending=True)

    ## Color Gradient

    norm = np.linspace(0, 1, len(values_df))
    colors_gradient = [
    f"rgb({int(182 - 182 * x)}, {int(217 - 143 * x)}, {int(168 - 56 * x)})"
    for x in norm
    ]

    tag_values_bar = go.Figure(
        go.Bar(
            y=values_df['Tag Value In Policy'],
            x=values_df['Usage Quantity'],
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


    #### Tags By SKU Heatmap
    heat_map_query = build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, 
                                               tag_filter=tag_filter, start_date=start_date, end_date=end_date, 
                                               product_category=product_category, 
                                               tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, 
                                               final_agg_query=HEATMAP_QUERY)
    
    heat_map_df = system_query_manager.execute_query_to_df(heat_map_query).sort_values(by='Usage Quantity', ascending=True) 
    heat_pivot = heat_map_df.pivot_table(values='Usage Quantity', index='Tag', columns='Product', fill_value=0)
    #heat_pivot['Total'] = heat_pivot.sum(axis=1)

    heat_map_fig = go.Figure(data=go.Heatmap(
                   z=heat_pivot.values,  # Provide the values from the pivot table
                   x=heat_pivot.columns,  # Set the x-axis to Product names
                   y=heat_pivot.index,    # Set the y-axis to Tag names
                   colorscale=colors_gradient,
                   hoverongaps = False))


    heat_map_fig.update_layout(
        title='Heatmap of Usage Quantity by Product and Tag',
        xaxis_title='Product',
        yaxis_title='Tag'
    )

    heat_map_fig.update_layout(ChartFormats.common_chart_layout())

    return fig, matched_fig, unmatched_fig, tag_values_bar, heat_map_fig



if __name__ == '__main__':

    app.run_server(debug=True)
