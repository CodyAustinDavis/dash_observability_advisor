CREATE TABLE IF NOT EXISTS app_sku_discounts
(
    discount_id BIGINT GENERATED BY DEFAULT AS IDENTITY,
    sku_name STRING,
    product_category STRING,
    discount_percent STRING,
    discounted_rate STRING,
    update_timestamp TIMESTAMP
)
TBLPROPERTIES('delta.feature.allowColumnDefaults' = 'supported');


CREATE MATERIALIZED VIEW IF NOT EXISTS clean_usage
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
  c.driver_node_type,
  c.worker_node_type,
  c.min_autoscale_workers,
  c.max_autoscale_workers,
  c.auto_termination_minutes,
  c.worker_count,
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
  )

-- Final Select
SELECT 
u.*,
c.*,
u.workspace_id AS clean_workspace_id,
  sku_price*usage_quantity AS Dollar_DBUs_List,
-- Clean up cluster / warehouse ids from all places
COALESCE(c.cluster_id, u.usage_metadata.cluster_id, u.usage_metadata.warehouse_id) AS clean_cluster_id,
COALESCE(c.job_or_pipeline_id, u.usage_metadata.job_id, u.usage_metadata.dlt_pipeline_id, 'Adhoc Or Serving') AS clean_job_or_pipeline_id,
-- Compute Type
u.identity_metadata.run_as::string AS UsageOwner,
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
;