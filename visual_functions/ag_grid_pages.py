
def get_adhoc_ag_grid_column_defs(tag_keys):
    
    updated_column_defs = [
        {'headerCheckboxSelection': True, 'checkboxSelection': True, 'headerCheckboxSelectionFilteredOnly': True, 'width': 50, 'suppressSizeToFit': True},
        {'headerName': 'Cluster ID', 'field': 'cluster_id', 'editable': False, 'width': 100, 'suppressSizeToFit': True},
        {'headerName': 'Cluster Name', 'field': 'cluster_name', 'editable': False, 'suppressSizeToFit': True},
        {'headerName': 'Policy Status', 'field': 'is_tag_policy_match', 'editable': False,'width': 100, 'suppressSizeToFit': True},
        {'headerName': 'Missing Policy Tags', 'field': 'missing_tags', 'editable': False,  'suppressSizeToFit': True},
        {'headerName': 'Tags', 'field': 'tags', 'editable': False, 'suppressSizeToFit': True},
        {'headerName': 'Add Policy Key', 'field': 'input_policy_key', 'editable': True, 'cellStyle': {'backgroundColor': 'rgba(111, 171, 208, 0.9)'}, 'suppressSizeToFit': True},
        # 'cellEditor': 'agSelectCellEditor', 'cellEditorParams': {'values': [option.get('value') for option in tag_keys]}},
        {'headerName': 'Add Policy Value', 'field': 'input_policy_value', 'editable': True, 'cellStyle': {'backgroundColor': 'rgba(111, 171, 208, 0.9)'},'suppressSizeToFit': True},
        {'headerName': 'Usage Amount', 'field': 'Dollar_DBUs_List', 'editable': False, 'suppressSizeToFit': True,
         'cellRenderer': 'BarGuage', 'guageColor': 'rgba(111, 171, 208, 0.7)'},
        {'headerName': 'T7 Usage', 'field': 'T7_Usage', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'BarGuage', 'guageColor': 'rgba(172, 213, 180, 0.6)'},
        {'headerName': 'T30 Usage', 'field': 'T30_Usage', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'BarGuage', 'guageColor': 'rgba(172, 213, 180, 0.6)'},
        {'headerName': 'T90 Usage', 'field': 'T90_Usage', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'BarGuage', 'guageColor': 'rgba(172, 213, 180, 0.6)'},
        {'headerName': 'Resource Age', 'field': 'resource_age', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'HeatMap'},
        {'headerName': 'Days Since Last Use', 'field': 'days_since_last_use', 'editable': False, 'suppressSizeToFit': True, 'cellRenderer': 'HeatMap'},
        {'headerName': 'Product Type', 'field': 'product_type', 'editable': False, 'suppressSizeToFit': True},
        {'headerName': 'Workspace ID', 'field': 'workspace_id', 'editable': False, 'suppressSizeToFit': True},
        {'headerName': 'Account ID', 'field': 'account_id', 'editable': False, 'suppressSizeToFit': True},
        {'headerName': 'First Usage Date', 'field': 'first_usage_date', 'editable': False, 'suppressSizeToFit': True},
        {'headerName': 'Latest Usage Date', 'field': 'latest_usage_date', 'editable': False, 'suppressSizeToFit': True},
        {'headerName': 'Resource Owner', 'field': 'resource_owner', 'editable': False, 'suppressSizeToFit': True},
        {'headerName': 'Usage Quantity', 'field': 'usage_quantity', 'editable': False, 'suppressSizeToFit': True}
    ]

    return updated_column_defs