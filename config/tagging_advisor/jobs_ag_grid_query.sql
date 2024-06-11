
WITH base_clusters AS (
SELECT 
clean_cluster_id,
MAX(clean_job_or_pipeline_id) AS job_id,
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
FROM clean_usage_table
GROUP BY clean_cluster_id
)

SELECT
job_id,
collect_list(clean_cluster_id) AS cluster_ids,
MAX(cluster_name) AS cluster_name,
MAX(product_type) AS product_type,
MAX(workspace_id) AS workspace_id,
MAX(account_id) AS account_id,
last_value(tags) AS tags,
MAX(resource_owner) AS resource_owner,
SUM(usage_quantity) AS usage_quantity,
SUM(Dollar_DBUs_List) AS Dollar_DBUs_List,
SUM(T7_Usage) AS T7_Usage,
SUM(T30_Usage) AS T30_Usage,
SUM(T90_Usage) AS T90_Usage,
MIN(first_usage_date) AS first_usage_date,
MAX(latest_usage_date) AS latest_usage_date,
date_diff(DAY, MIN(first_usage_date) , getdate()) AS resource_age,
date_diff(DAY, MAX(latest_usage_date) , getdate()) AS days_since_last_use
FROM base_clusters 
WHERE product_type IN ('JOBS')
GROUP BY job_id;