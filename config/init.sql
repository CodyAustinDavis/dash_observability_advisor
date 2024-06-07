

CREATE MATERIALIZED VIEW IF NOT EXISTS clean_usage
TBLPROPERTIES('pipelines.autoOptimize.zOrderCols' = 'usage_start_time,billing_origin_product')
AS
WITH compute AS (
SELECT
  c.cluster_name,
  CASE
    WHEN c.cluster_name LIKE 'job-%' THEN 'Job Cluster'
    WHEN c.cluster_name LIKE 'dlt-%' THEN 'DLT Pipeline'
    ELSE 'Adhoc Cluster'
  END AS cluster_type,
  CASE
    WHEN c.cluster_name LIKE 'job-%' THEN regexp_extract(c.cluster_name, '^job-(\\d+)-')
    WHEN c.cluster_name LIKE 'dlt-%' THEN regexp_extract(c.cluster_name, '^dlt-(\\d+)-')
    ELSE NULL
  END AS job_or_pipeline_id,
  c.owned_by,
  c.cluster_id,
  c.create_time AS cluster_create_time,
  c.delete_time AS cluster_delete_time,
  CASE WHEN c.delete_time IS NOT NULL AND c.cluster_id IS NOT NULL THEN 1 ELSE 0 END AS IsClusterDeleted,
  c.tags AS cluster_tags,
  c.cluster_source
FROM system.compute.clusters c
QUALIFY (ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY change_time DESC) = 1)
),
 
px_all AS (
  SELECT DISTINCT
  sku_name,
  pricing.default AS unit_price,
  unit_price::decimal(10,3) AS sku_price
  FROM system.billing.list_prices 
  QUALIFY ROW_NUMBER() OVER (PARTITION BY sku_name ORDER BY price_start_time DESC) = 1
  ),

clean_usage AS (
-- Final Select
SELECT 
u.*,
c.*,
u.workspace_id AS clean_workspace_id,
sku_price*usage_quantity AS Dollar_DBUs_List,
-- Clean up cluster / warehouse ids from all places
COALESCE(c.cluster_id, u.usage_metadata.cluster_id) AS clean_cluster_id,
COALESCE(u.usage_metadata.warehouse_id) AS clean_warehouse_id,
COALESCE(u.usage_metadata.job_id, u.usage_metadata.dlt_pipeline_id, c.job_or_pipeline_id,  NULL) AS clean_job_or_pipeline_id,
map_zip_with(
  IFNULL(u.custom_tags, map()),
  IFNULL(c.cluster_tags, map()),
  (k, v1, v2) -> COALESCE(v1, v2)
) AS clean_tags,
-- Compute Type
COALESCE(u.identity_metadata.run_as::string, c.owned_by) AS clean_usage_owner,
--Job Type (DLT / Jobs / )
CASE WHEN u.sku_name LIKE ('%SERVERLESS%') OR u.product_features.is_serverless = 'true' THEN 'Serverless' ELSE 'Self-hosted' END AS IsServerless

FROM system.billing.usage AS u
INNER JOIN px_all AS px ON px.sku_name = u.sku_name
LEFT JOIN compute AS c
    ON (c.cluster_id = u.usage_metadata.cluster_id
   --     AND (   (c.job_or_pipeline_id = u.usage_metadata.job_id)
   --             OR (c.job_or_pipeline_id = u.usage_metadata.dlt_pipeline_id)
   --             OR (c.job_or_pipeline_id = 'Adhoc')
   --         )
    )
)

SELECT * FROM clean_usage;