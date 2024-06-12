SELECT 
clean_cluster_id AS cluster_id,
MAX(cluster_name) AS cluster_name,
MIN(IsTaggingMatch) AS is_tag_policy_match,
  flatten(collect_set(MatchedTagKeys)) AS tag_matches,
  flatten(collect_set(MissingTagKeys)) AS missing_tags,
MAX(billing_origin_product) AS product_type,
MAX(workspace_id) AS workspace_id,
MAX(account_id) AS account_id,
last_value(clean_tags) AS tags,
MAX(clean_usage_owner) AS resource_owner,
round(SUM(usage_quantity), 2) AS usage_quantity,
round(SUM(Dollar_DBUs_List), 2) AS Dollar_DBUs_List,
round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 7  THEN Dollar_DBUs_List END), 2) AS T7_Usage,
round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 30  THEN Dollar_DBUs_List END), 2) AS T30_Usage,
round(SUM(CASE WHEN date_diff(DAY, usage_date, getdate()) <= 90  THEN Dollar_DBUs_List END), 2) AS T90_Usage,
MIN(usage_date) AS first_usage_date,
MAX(usage_date) AS latest_usage_date,
date_diff(DAY, first_usage_date , getdate()) AS resource_age,
date_diff(DAY, latest_usage_date , getdate()) AS days_since_last_use
FROM filtered_result
WHERE 1=1 
AND billing_origin_product IN ('ALL_PURPOSE')
GROUP BY clean_cluster_id
ORDER BY Dollar_DBUs_List DESC