
"""
Author: Cody Austin Davis
Date: 5/25/2024
Description: Dash Apps for Tagging and Data Compute Observability

TO DO: 
1. Add databricsk SDK for authentication and iteraction with db
6. Create settings page that lets a user input the workspace name and select the warehouse to execute against. (OAuth)
7. Create LLM to generate visuals for system tables upon request / Alerts
8. Add Top N filter with default 100 for Usage Grids
9. Need to add Warehouse Name and owner once Warehouses system tables is available
10. Add ability to filter by actual tag key / value from the usage - this helps apply policies to subsets of tags as well


BUGS: 
1. Make filters update based on other selected filter values
2. AG Grid Cannot handle deleting tag policies and having it show up properly. I think this is a spark bug - UC is right but the query is now
3. For the Usage AG Grids, optimize data retrieval by not calling back to the database each time the buttons are pressed, just remove the changes in client side

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
from threading import Thread
from data_functions.utils import *


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

## Load Visual Specific Query Parts
AGG_QUERY = read_sql_file("./config/tagging_advisor/tag_date_agg_query.sql")
MATCHED_IND_QUERY = read_sql_file("./config/tagging_advisor/matched_indicator_query.sql")
TAG_VALUES_QUERY = read_sql_file("./config/tagging_advisor/tag_values_query.sql")
HEATMAP_QUERY = read_sql_file("./config/tagging_advisor/tag_sku_heatmap_query.sql")



# Function to refresh materialized view
def refresh_materialized_view():
    conn = system_query_manager.get_engine().connect()
    cursor = conn.cursor()
    cursor.execute("REFRESH MATERIALIZED VIEW clean_usage")
    conn.commit()
    cursor.close()
    conn.close()

# Function to run the SQL query in a separate thread

def run_query_async():
    thread = Thread(target=refresh_materialized_view)
    thread.start()
    return thread


# Initialize a global variable to keep track of the thread
query_thread = None

# Define the layout of the main app

app.layout = dbc.Container([
    dcc.Store(id='sidebar-state', data={'is_open': True}),
    dbc.Row([
        dbc.Col([
            dbc.Button("☰", id="toggle-button", n_clicks=0, className='toggle-button'),
            dbc.Collapse(
                dbc.Nav(
                    [
                        html.Div([  # Container for the logo and possibly other header elements
                            html.Img(id='sidebar-logo', src='/assets/app_logo.png', style={'width': '100%', 'height': 'auto'}),
                        ]),
                        dbc.NavLink("Tags", href="/tag-manager", id="tab-1-link"),
                        dbc.NavLink("Alerts", href="/alert-manager", id="tab-2-link"),
                        dbc.NavLink("Contracts", href="/contract-manager", id="tab-3-link"),
                        dbc.NavLink("Settings", href="/settings", id="tab-4-link"),
                    ],
                    vertical=True,
                    pills=True,
                    className="sidebar"
                ),
                id="sidebar",
                is_open=True,
            )
        ], width={"size": 1, "offset": 0}, id="sidebar-col", className="sidebar-col"),
        dbc.Col([
            dcc.Location(id='url', refresh=False),
            html.Div(id='tab-content', className="main-content")
        ], id="main-content-col", width=11)
    ])
], fluid=True, className="app-container")




@app.callback(
    [Output("sidebar", "is_open"),
     Output("sidebar-col", "width"),
     Output("main-content-col", "width"),
     Output("toggle-button", "children"),
     Output("sidebar-logo", "style"),
     Output("sidebar-state", "data")],
    [Input("toggle-button", "n_clicks")],
    [State("sidebar-state", "data")]
)
def toggle_sidebar(n, sidebar_state):
    if n:
        is_open = not sidebar_state['is_open']
        sidebar_state['is_open'] = is_open
        sidebar_width = 1 if is_open else 0
        main_content_width = 11 if is_open else 12
        button_text = "☰" if is_open else "☰"
        logo_style = {'width': '100%', 'height': 'auto'} if is_open else {'display': 'none'}
        return is_open, {"size": sidebar_width, "offset": 0}, main_content_width, button_text, logo_style, sidebar_state
    return sidebar_state['is_open'], {"size": 1, "offset": 0}, 11, "☰", {'width': '100%', 'height': 'auto'}, sidebar_state



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
    query = build_tag_query_from_params(tag_filter=tag_filter, start_date=start_date, end_date=end_date, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=AGG_QUERY)
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
    ind_query = build_tag_query_from_params(tag_filter=tag_filter, start_date=start_date, end_date=end_date, product_category=product_category, tag_policies=tag_policies, tag_keys=tag_keys, tag_values=tag_values, final_agg_query=MATCHED_IND_QUERY)
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
    values_query = build_tag_query_from_params(
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
    heat_map_query = build_tag_query_from_params(
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

            distinct_tag_policies = pd.read_sql(session.query(TagPolicies.tag_policy_name).distinct().statement, con=system_engine)
            distinct_tag_keys = pd.read_sql(text("""
                                    SELECT tag_key FROM main.dash_observability_advisor.app_tag_policies
                                    QUALIFY ROW_NUMBER() OVER (PARTITION BY tag_key ORDER BY update_timestamp DESC) = 1"""), con=system_engine)
            distinct_tag_values = pd.read_sql(session.query(TagPolicies.tag_value).distinct().statement, con=system_engine)

            tag_policy_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_policies['tag_policy_name']]
            tag_key_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_keys['tag_key']]
            tag_value_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_values['tag_value']]
            
            print(f"NEW TAG KEYS DELETE INNER LOOP: {tag_key_filter}")

        ## DEBUG
        print(f"NEW TAG DELETEKEYSSS: {tag_key_filter}")

        return updated_row_data, get_tag_policies_grid_data().to_dict('records'), dash.no_update, dash.no_update, dash.no_update, tag_policy_filter, tag_key_filter, tag_value_filter, get_adhoc_ag_grid_column_defs(tag_key_filter), tag_key_filter

    ##### Handle Clear Button Press
    elif triggered_id == 'tag-policy-clear-btn' and clear_clicks:
        clear_loading_content = html.Button('Clear Policy Changes', id='tag-policy-clear-btn', n_clicks=0, style={'margin-bottom': '10px'}, className = 'prettier-button')
        return [], get_tag_policies_grid_data().to_dict('records'), {'display': 'none'}, dash.no_update, clear_loading_content, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update # Clear changes and reload data
    
    
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
            
         # Fetch distinct tag policy names within the context manager
        with QueryManager.session_scope(system_engine) as session:

            distinct_tag_policies = pd.read_sql(session.query(TagPolicies.tag_policy_name).distinct().statement, con=system_engine)
            distinct_tag_keys = pd.read_sql(text("""
                                    SELECT tag_key FROM main.dash_observability_advisor.app_tag_policies
                                    QUALIFY ROW_NUMBER() OVER (PARTITION BY tag_key ORDER BY update_timestamp DESC) = 1
                                                 """), con=system_engine)
            distinct_tag_values = pd.read_sql(session.query(TagPolicies.tag_value).distinct().statement, con=system_engine)

            tag_policy_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_policies['tag_policy_name']]
            tag_key_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_keys['tag_key']]
            tag_value_filter = [{'label': name if name is not None else 'None', 'value': name if name is not None else 'None'} for name in distinct_tag_values['tag_value']]

    
            return [], get_tag_policies_grid_data().to_dict('records'), {'display': 'none'}, save_loading_content, dash.no_update, tag_policy_filter, tag_key_filter, tag_value_filter, get_adhoc_ag_grid_column_defs(tag_key_filter), tag_key_filter
        

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
   [Input('start-date-picker', 'date'),
    Input('end-date-picker', 'date'),
    Input('tag-filter-dropdown', 'value'),
    Input('tag-policy-dropdown', 'value'),
    Input('tag-policy-key-dropdown', 'value'),
    Input('tag-policy-value-dropdown', 'value'),
    ### Actual AG Grid State
    Input('usage-adhoc-save-btn', 'n_clicks'), 
     Input('usage-adhoc-clear-btn', 'n_clicks'),
     Input('adhoc-usage-ag-grid', 'cellValueChanged'),
     Input('top-n-adhoc', 'value')],
    [State('adhoc-usage-ag-grid', 'rowData'),
     State('adhoc-ag-grid-store', 'data'),
     State('adhoc-ag-grid-original-data', 'data'),
     State('adhoc-usage-ag-grid', 'selectedRows')]
)
def update_adhoc_grid_data(start_date, end_date, tag_filter, tag_policies, tag_keys, tag_values, 
                           save_clicks, clear_clicks, cell_change, top_n,
                           row_data, changes, original_data, selected_rows):

    
    ### Synchronously figure out what action is happening and run the approprite logic. 
    ### This single-callback method ensure that no strange operation order can happen. Only one can happen at once. 
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]


    ##### Handle Changes to top N filter
    if triggered_id == 'top-n-adhoc' and top_n:
       
       updated_data = get_adhoc_clusters_grid_data(
        start_date=start_date,
        end_date=end_date,
        tag_filter=tag_filter,
        tag_policies=tag_policies,
        tag_keys=tag_keys,
        tag_values=tag_values,
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
    if triggered_id == 'usage-adhoc-save-btn' and save_clicks:

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
   [Input('start-date-picker', 'date'),
    Input('end-date-picker', 'date'),
    Input('tag-filter-dropdown', 'value'),
    Input('tag-policy-dropdown', 'value'),
    Input('tag-policy-key-dropdown', 'value'),
    Input('tag-policy-value-dropdown', 'value'),
    ### Actual AG Grid State
    Input('usage-jobs-save-btn', 'n_clicks'), 
     Input('usage-jobs-clear-btn', 'n_clicks'),
     Input('jobs-usage-ag-grid', 'cellValueChanged'),
     Input('top-n-jobs', 'value')], 
    [State('jobs-usage-ag-grid', 'rowData'),
     State('jobs-ag-grid-store', 'data'),
     State('jobs-ag-grid-original-data', 'data'),
     State('jobs-usage-ag-grid', 'selectedRows')]
)
def update_jobs_grid_data(start_date, end_date, tag_filter, tag_policies, tag_keys, tag_values, 
                           save_clicks, clear_clicks, cell_change, top_n,
                           row_data, changes, original_data, selected_rows):

    
    ### Synchronously figure out what action is happening and run the approprite logic. 
    ## This single-callback method ensure that no strange operation order can happen. Only one can happen at once. 
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]


    ##### Handle Changes to top N filter
    if triggered_id == 'top-n-jobs' and top_n:
       
       updated_data = get_jobs_clusters_grid_data(
        start_date=start_date,
        end_date=end_date,
        tag_filter=tag_filter,
        tag_policies=tag_policies,
        tag_keys=tag_keys,
        tag_values=tag_values,
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
   [Input('start-date-picker', 'date'),
    Input('end-date-picker', 'date'),
    Input('tag-filter-dropdown', 'value'),
    Input('tag-policy-dropdown', 'value'),
    Input('tag-policy-key-dropdown', 'value'),
    Input('tag-policy-value-dropdown', 'value'),
    ### Actual AG Grid State
    Input('usage-sql-save-btn', 'n_clicks'), 
     Input('usage-sql-clear-btn', 'n_clicks'),
     Input('sql-usage-ag-grid', 'cellValueChanged'),
     Input('top-n-sql', 'value')], 
    [State('sql-usage-ag-grid', 'rowData'),
     State('sql-ag-grid-store', 'data'),
     State('sql-ag-grid-original-data', 'data'),
     State('sql-usage-ag-grid', 'selectedRows')]
)
def update_sql_grid_data(start_date, end_date, tag_filter, tag_policies, tag_keys, tag_values, 
                           save_clicks, clear_clicks, cell_change, top_n,
                           row_data, changes, original_data, selected_rows):

    
    ### Synchronously figure out what action is happening and run the approprite logic. 
    ## This single-callback method ensure that no strange operation order can happen. Only one can happen at once. 
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]


    ##### Handle Changes to top N filter
    if triggered_id == 'top-n-sql' and top_n:
       
       updated_data = get_sql_clusters_grid_data(
        start_date=start_date,
        end_date=end_date,
        tag_filter=tag_filter,
        tag_policies=tag_policies,
        tag_keys=tag_keys,
        tag_values=tag_values,
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

        updated_data_after_save = get_compute_tagged_grid_data().to_dict('records')
            
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
        updated_data = get_compute_tagged_grid_data().to_dict('records')
        return updated_data
    
    if tags:
        current_state = rowData

        return current_state
    
    if reloadTrigger:
        return get_compute_tagged_grid_data().to_dict('records')

    
    return dash.no_update


##### END APP #####
if __name__ == '__main__':
    app.run_server(debug=True)
