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


def render_alert_manager_page():
    return dbc.Container([
        dbc.Row([
                dbc.Col([
                    html.H1("Alert Manager", style={'color': '#002147'}),  # A specific shade of blue
                ], width=12),
        ]),
        dbc.Row([
                dbc.Col([
                ], width=12),
        ])
    ]
    , fluid=True)


