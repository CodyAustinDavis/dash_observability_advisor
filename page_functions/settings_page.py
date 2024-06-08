import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from data_functions.backend_database import (
    QueryManager, 
    get_product_category_filter_tuples,
    get_start_ts_filter_min,
    get_end_ts_filter_max,
    get_tag_policy_name_filter,
    get_tag_policy_key_filter,
    get_tag_policy_value_filter
)
from sqlalchemy import create_engine, text


class SettingsPageManager():

    def __init__(self, system_query_manager):
        self.system_query_manager = system_query_manager

    # Function to refresh materialized view
    def refresh_materialized_view(self):
        try: 
            connection = self.system_query_manager.get_engine().connect()
            connection.execute(text("REFRESH MATERIALIZED VIEW clean_usage"))
            connection.commit()
            connection.close()
            return "Materialized View Refreshed!"
        
        except Exception as e:
            return f"FAILED TO REFRESH DATA MODEL: {str(e)}"


    def set_materialized_view_schedule(self, cron_schedule):
        connection = self.system_query_manager.get_engine().connect()
        msg = "Not Refreshed"

        if cron_schedule == 'None':
            try: 
                connection.execute(text(f"""ALTER MATERIALIZED VIEW clean_usage
                                    DROP SCHEDULE"""))
                connection.commit()
                connection.close()
                msg = "Schedule Dropped"

            except Exception as ed:
                msg = f"Failed to unset or clear schedule: {str(ed)}"

            finally:
                return msg

        
        ## alter or set schedule
        try:
            connection.execute(text(f"""ALTER MATERIALIZED VIEW clean_usage
                                ALTER SCHEDULE CRON '{cron_schedule}'"""))
            connection.commit()
            connection.close()
            msg = "Refresh Schedule Updated - Altered Existing Schedule!"

        except Exception as e:
            ## Try adding if none exists
            try: 
                connection.execute(text("""ALTER MATERIALIZED VIEW clean_usage
                                ADD SCHEDULE CRON '{cron_schedule}'"""))
                connection.commit()
                connection.close()
                msg = "Refresh Schedule Updated - Added New Schedule!"

            except Exception as e2:

                msg = f"FAILED TO UPDATE REFRESH SCHEDDULE with schedule {cron_schedule}, tried to set new instead of alter: {str(e2)}"

        finally:
            return msg



### Different page
def render_settings_page():


    layout =  dbc.Container([
        dbc.Row([
                dbc.Col([
                    html.H1("Settings", style={'color': '#002147'}),  # A specific shade of blue
                ], width=8),
        ]),
        html.Div(className='border-top'),
        dbc.Row([
                dbc.Col([
                ], width=12),
        ]),
        #### Materialized View Schedule Setting - Schedule (if any), Update Schedule, Refresh Now
        dbc.Row([
                 dbc.Col([
                    html.P("Data Model Materialized View Schedule:", style={'color': '#002147'}),  # A specific shade of blue
                ], width=2),
                dbc.Col(
                dbc.Input(
                    id="mv-cron-schedule",
                    placeholder="Cron Schedule for Data Model",
                    type="text",
                    value="",
                    size="lg"
                ),
                width=4
                ),
                dbc.Col([
                    dcc.Loading(
                                id="loading-set-mv-schedule",
                                type="default",
                                color= '#002147', 
                                children=[html.Button('Update Schedule', id='set-mv-schedule', n_clicks=0, className = 'prettier-button'),
                                    html.Div(id="update-schedule-output", style={'margin-top': '10px'})
                                ]
                            )
                ], width=2),
                dbc.Col([
                    dcc.Loading(
                                id="loading-refresh-mv-now",
                                type="default",
                                color= '#002147', 
                                children=[
                                    html.Button('Refresh Data', id='refresh-mv', n_clicks=0, className = 'prettier-button'),
                                    html.Div(id="refresh-mv-output", style={'margin-top': '10px'})
                                    ])
                ], width=2),
        ]),
        html.Div(className='border-top'),
        dbc.Row([
            dbc.Col([
                    html.P("App Cache Settings:", style={'color': '#002147'}),  # A specific shade of blue
                ], width=2),
            dbc.Col([
                html.Button("Clear Cache", id="clear-cache-button", className="prettier-button-warn"),
                dcc.Loading(id="loading-clear-cache", children=[
                    html.Div(id="cache-clear-output", style={'margin-top': '10px'})
                ], type="default")
            ])
        ]),
        html.Div(className='border-top'),
            
    ], fluid=True)


    return layout

