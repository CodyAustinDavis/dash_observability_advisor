
"""
Author: Cody Austin Davis
Date: 5/25/2024
Description: Dash Apps for Tagging and Data Compute Observability

TO DO: 
1. Add databricsk SDK for authentication and iteraction with db
2. Create SQL Alchemy backend and engine - DONE
4. Create AG Grid for tags
5. Create job button for syncing tags with app (tags jobs that arent tagged yet)
6. Create settings page that lets a user input the workspace name and select the warehouse to execute against. (OAuth)
7. Create LLM to generate visuals for system tables upon request


BUGS: 
1. Make filters update based on other selected filter values

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
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sqlalchemy import bindparam


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
TAG_QUERY_1 = read_sql_file("./config/tagging_advisor/base_tag_query_1.sql") ## Where you add selected tags
TAG_QUERY_2 = read_sql_file("./config/tagging_advisor/base_tag_query_2.sql")
TAG_QUERY_3 = read_sql_file("./config/tagging_advisor/base_tag_query_3.sql") ## Where you filter the rest of the paramters
TAG_QUERY_4 = read_sql_file("./config/tagging_advisor/base_tag_query_4.sql") ## Where you select the final data frame

## Load Visual Specific Query Parts
AGG_QUERY = read_sql_file("./config/tagging_advisor/tag_date_agg_query.sql")
MATCHED_IND_QUERY = read_sql_file("./config/tagging_advisor/matched_indicator_query.sql")
TAG_VALUES_QUERY = read_sql_file("./config/tagging_advisor/tag_values_query.sql")
HEATMAP_QUERY = read_sql_file("./config/tagging_advisor/tag_sku_heatmap_query.sql")



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
                dbc.NavLink("Tagging Manager", href="/tag-manager", id="tab-1-link"),
                dbc.NavLink("Alert Manager", href="/alert-manager", id="tab-2-link"),
                dbc.NavLink("Contract Manager", href="/contract-manager", id="tab-3-link"),
                dbc.NavLink("Settings", href="/settings", id="tab-4-link"),
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
    return pathname == "/tag-manager", pathname == "/alert-manager", pathname == "/contract-manager", pathname == "/settings", 


# Callback to update the content based on tab selection
@app.callback(
    Output('tab-content', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):

    ## TO DO: Add authentication screen
    ## Can one of these be called from an LLM to create visuals???

    if pathname == "/tag-manager":
        return render_tagging_advisor_page()
    elif pathname == "/alert-manager":
        return render_alert_manager_page()
    elif pathname == "/contract-manager":
        return render_contract_manager_page()
    elif pathname == "/settings":
        return render_settings_page()
    else:
        return render_tagging_advisor_page()



##### Update Usage Chart


@app.callback(
    Output('usage-by-match-chart', 'figure'),
    Output('matched-usage-ind', 'figure'),
    Output('unmatched-usage-ind', 'figure'),
    Output('usage-by-tag-value-chart', 'figure'),
    Output('usage-heatmap', 'figure'),
    Output('percent-match-ind', 'figure'),
    Output('total-usage-ind', 'figure'),
    Input('update-params-button', 'n_clicks'),
    State('tag-filter-dropdown', 'value'),
    State('start-date-picker', 'date'),
    State('end-date-picker', 'date'),
    State('product-category-dropdown', 'value'), ## Tuple
    State('tag-policy-dropdown', 'value'), ## Tuple
    State('tag-policy-key-dropdown', 'value'), ## Tuple
    State('tag-policy-value-dropdown', 'value'), ## Tuple
)
def update_usage_by_match(n_clicks, tag_filter, start_date, end_date, product_category, tag_policies, tag_keys, tag_values):

    # Define the color mapping for compliance
    color_map = {
        'In Policy': '#097969',
        'Not Matched To Tag Policy': '#002147'
    }

    # Convert the result to a DataFrame
    query = build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, tag_filter=tag_filter, start_date=start_date, end_date=end_date, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=AGG_QUERY)
    # Convert the result to a DataFrame
    df = system_query_manager.execute_query_to_df(query)

    # Create a Plotly Express line chart
    fig = px.bar(df, x='Usage_Date', y='Usage Amount', 
                    title='Daily Usage By Tag Policy Match',
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
    ind_query = build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, tag_filter=tag_filter, start_date=start_date, end_date=end_date, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=MATCHED_IND_QUERY)
    ind_df = system_query_manager.execute_query_to_df(ind_query)
    matched_value = safe_round(ind_df['Matched Usage Amount'][0], 0)
    not_matched_value = safe_round(ind_df['Not Matched Usage Amount'][0], 0)


    #### Percent Matched Usage Indicator
    percent_match_fig = go.Figure(go.Indicator(
                            mode="number",
                            value=safe_divide(matched_value, safe_add(matched_value, not_matched_value)) ,
                            title={"text": "% Matched Usage", 'font': {'size': 24}},
                            number={'font': {'size': 42, 'color': "#097969"}, 'valueformat': ',.1%'},
                            domain = {'x': [0, 1], 'y': [0, 0.9]}  # Adjust domain to fit elements
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
                            number={'font': {'size': 42, 'color': "#097969"}, 'valueformat': '$,'},
                            domain = {'x': [0, 1], 'y': [0, 0.9]}  # Adjust domain to fit elements
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
                            title={"text": "Not Matched Usage", 'font': {'size': 24}},
                            number={'font': {'size': 24, 'color': '#8B0000'}, 'valueformat': '$,'},
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
    
    values_df = system_query_manager.execute_query_to_df(values_query).sort_values(by='Usage Amount', ascending=True)

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


    #### Tags By SKU Heatmap
    heat_map_query = build_tag_query_from_params(TAG_QUERY_1, TAG_QUERY_2, TAG_QUERY_3, 
                                               tag_filter=tag_filter, start_date=start_date, end_date=end_date, 
                                               product_category=product_category, 
                                               tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, 
                                               final_agg_query=HEATMAP_QUERY)
    
    heat_map_df = system_query_manager.execute_query_to_df(heat_map_query).sort_values(by='Usage Amount', ascending=True) 
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

    return fig, matched_fig, unmatched_fig, tag_values_bar, heat_map_fig, percent_match_fig, total_usage_fig

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
     Output('tag-policy-value-dropdown', 'options')],
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

        # Helper function to mark changed rows
    # Helper function to mark changed rows
    def mark_changed_rows(data, changes):
        for row in data:
            row['isChanged'] = 1 if any(row['tag_policy_id'] == change['tag_policy_id'] for change in changes) else 0
        return data
    

    # Get the grid data and mark changed rows
    grid_data = get_tag_policies_grid_data().to_dict('records')
    if changes:
        grid_data = mark_changed_rows(grid_data, changes)


    # Handle adding a new row
    if triggered_id == 'add-policy-row-btn' and add_row_clicks > 0:
        new_row = {
            'tag_policy_id': None,  # Will be generated by the database
            'tag_policy_name': '',
            'tag_key': '',
            'tag_value': '',
            'update_timestamp': datetime.now()
        }
        row_data.append(new_row)
        return dash.no_update, row_data, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Handle removing selected rows
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

            distinct_tag_policies = pd.read_sql(session.query(TagPolicies.tag_policy_name).distinct().statement, con=system_engine)
            distinct_tag_keys = pd.read_sql(session.query(TagPolicies.tag_key).distinct().statement, con=system_engine)
            distinct_tag_values = pd.read_sql(session.query(TagPolicies.tag_value).distinct().statement, con=system_engine)

            tag_policy_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_policies['tag_policy_name']]
            tag_key_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_keys['tag_key']]
            tag_value_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_values['tag_value']]


        return updated_row_data, get_tag_policies_grid_data().to_dict('records'), dash.no_update, dash.no_update, dash.no_update, tag_policy_filter, tag_key_filter, tag_value_filter

    ## Handle Local Updates to Store (do NOT clear store)
    # Accumulate changes for new records
    def group_changes_by_row(changes):

        if changes:

            grouped_changes = {}
            for change in changes:
                row_index = change['row_index']
                if row_index not in grouped_changes:
                    grouped_changes[row_index] = change.copy()
                else:
                    for k, v in change.items():
                        if v:
                            grouped_changes[row_index][k] = v

            return list(grouped_changes.values())
        
        else:
            return []
        


    # Combine changes by row index
    grouped_changes = []

    if changes:
        grouped_changes = group_changes_by_row(changes)

    # Handle saving changes
    if triggered_id == 'tag-policy-save-btn' and save_clicks:

        connection = system_query_manager.get_engine().connect()
        save_loading_content = html.Button('Save Policy Changes', id='tag-policy-save-btn', n_clicks=0, style={'margin-bottom': '10px'}, className='prettier-button')

        print(f"GROUPED CHANGES DATA: {grouped_changes}")
        print(changes)

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
                                tag_key = :tag_key,
                                tag_value = :tag_value,
                                update_timestamp = NOW()
                            WHERE tag_policy_id = :tag_policy_id
                        """)
                        connection.execute(update_query, parameters={
                            'tag_policy_name': change['tag_policy_name'],
                            'tag_key': change['tag_key'],
                            'tag_value': change['tag_value'],
                            'tag_policy_id': record_id
                        })
                    else:
                        # Insert new record
                        if not change.get('tag_policy_name') or not change.get('tag_key'):
                            raise ValueError("Missing required fields: 'tag_policy_name' or 'tag_key'")

                        insert_params = {k: v for k, v in change.items() if k in ['tag_policy_name', 'tag_key', 'tag_value']}
                        print(f"INSERT PARAMS: {insert_params}")  # Debug statement

                        insert_query = text("""
                            INSERT INTO app_tag_policies (tag_policy_name, tag_key, tag_value, update_timestamp)
                            VALUES (:tag_policy_name, :tag_key, :tag_value, NOW())
                        """)
                        connection.execute(insert_query, parameters= {'tag_policy_name':insert_params['tag_policy_name'],
                                        'tag_key':insert_params['tag_key'],
                                        'tag_value': insert_params['tag_value']})

            except Exception as e:
                print(f"Error during save with changes: {changes}")  # Debug error
                raise e
            finally:
                connection.close()

        else:
            pass
            
         # Fetch distinct tag policy names within the context manager
        with QueryManager.session_scope(system_engine) as session:

            distinct_tag_policies = pd.read_sql(session.query(TagPolicies.tag_policy_name).distinct().statement, con=system_engine)
            distinct_tag_keys = pd.read_sql(session.query(TagPolicies.tag_key).distinct().statement, con=system_engine)
            distinct_tag_values = pd.read_sql(session.query(TagPolicies.tag_value).distinct().statement, con=system_engine)

            tag_policy_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_policies['tag_policy_name']]
            tag_key_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_keys['tag_key']]
            tag_value_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_values['tag_value']]

        
        return [], get_tag_policies_grid_data().to_dict('records'), {'display': 'none'}, save_loading_content, dash.no_update, tag_policy_filter, tag_key_filter, tag_value_filter
        
    ### Handle Clear Button Press
    elif triggered_id == 'tag-policy-clear-btn' and clear_clicks:
        clear_loading_content = html.Button('Clear Policy Changes', id='tag-policy-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
        return [], get_tag_policies_grid_data().to_dict('records'), {'display': 'none'}, dash.no_update, clear_loading_content, dash.no_update, dash.no_update, dash.no_update # Clear changes and reload data
    
    
    # Handle cell change
    elif triggered_id == 'tag-policy-ag-grid' and cell_change:
        if changes is None:
            changes = []
        change_data = cell_change[0]['data']
        row_index = cell_change[0]['rowIndex']
        # Ensure the change data includes the row index
        change_data['row_index'] = row_index
        changes.append(change_data)
        row_data = mark_changed_rows(row_data, changes)
        return changes, row_data, {'display': 'block', 'color': 'yellow', 'font-weight': 'bold'}, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update  # No action taken


if __name__ == '__main__':

    app.run_server(debug=True)
