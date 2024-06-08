-- DLT pipelines (ST/MVs) will be billed partially as SQL usage BUT not have a warehouse_id, but they will have a DLT pipeline Id

SELECT 
clean_warehouse_id,
MAX(cluster_name) AS cluster_name,
MAX(billing_origin_product) AS product_type,
MAX(workspace_id) AS workspace_id,
MAX(account_id) AS account_id,
last_value(clean_tags) AS tags,
MAX(clean_usage_owner) AS resource_owner,
SUM(usage_quantity) AS usage_quantity,
SUM(Dollar_DBUs_List) AS Dollar_DBUs_List,
SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 7  THEN Dollar_DBUs_List END) AS T7_Usage,
SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 30  THEN Dollar_DBUs_List END) AS T30_Usage,
SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 90  THEN Dollar_DBUs_List END) AS T90_Usage,
MIN(usage_date) AS first_usage_date,
MAX(usage_date) AS latest_usage_date,
date_diff(DAY, first_usage_date , getdate()) AS resource_age,
date_diff(DAY, latest_usage_date , getdate()) AS days_since_last_use
FROM clean_usage
WHERE billing_origin_product IN ('SQL')
AND clean_warehouse_id IS NOT NULL
GROUP BY clean_warehouse_id
