),

combined_cluster_tags AS (
        SELECT
        DISTINCT compute_asset_id, billing_origin_product, TagKey, TagValue, 1 AS IsUsageTag
    FROM 
        (SELECT COALESCE(clean_job_or_pipeline_id, clean_warehouse_id, clean_cluster_id) AS compute_asset_id, billing_origin_product, clean_tags FROM filtered_usage)
        LATERAL VIEW EXPLODE(clean_tags) AS TagKey, TagValue
        WHERE compute_asset_id IS NOT NULL

    UNION 
        SELECT compute_asset_id AS compute_asset_id, compute_asset_type AS billing_origin_product, tag_key AS TagKey, tag_value AS TagValue, 1 AS IsUsageTag
        FROM app_compute_tags
),

tag_potential_matches AS (

    SELECT  
    user_tags.compute_asset_id,
    user_tags.billing_origin_product,
    (SELECT COUNT(0) FROM active_tags) AS TotalPolicyTags,
    SUM(COALESCE(IsPolicyTag, 0)) AS NumberOfMatchedKeys,
    COUNT(DISTINCT tag_value) AS NumberOfMatchedValues,
    CASE
        WHEN NumberOfMatchedKeys >= TotalPolicyTags THEN 'In Policy'
        ELSE 'Not Matched To Tag Policy'
    END AS IsTaggingMatch,
    collect_set(CONCAT(TagKey, COALESCE(CONCAT(': ', TagKey), ''))) AS TagCombos, --TagCombo from tag policies
    collect_set(CASE WHEN IsPolicyTag = 1 THEN TagKey END) AS MatchingTagKeys,
    collect_set(CASE WHEN IsPolicyTag = 1 THEN TagValue END) AS MatchingTagValues,
    collect_set(CONCAT(TagKey, COALESCE(CONCAT(': ', TagValue), ''))) AS updated_tags

    FROM combined_cluster_tags AS user_tags
    LEFT JOIN (SELECT *, CONCAT(tag_key, COALESCE(CONCAT(': ', tag_value), '')) AS TagCombo FROM active_tags) p
    ON 
        user_tags.TagKey = p.tag_key
        AND (p.tag_value IS NULL OR p.tag_value = "" OR user_tags.TagValue = p.tag_value)
    GROUP BY 
    user_tags.compute_asset_id,
    user_tags.billing_origin_product
),

unmatched_policies AS (
    SELECT 
        a.compute_asset_id,
        p.tag_key AS UnmatchedPolicyKey
    FROM (
        SELECT DISTINCT compute_asset_id, billing_origin_product 
        FROM combined_cluster_tags
    ) a
    CROSS JOIN active_tags p
    LEFT JOIN combined_cluster_tags u
    ON a.compute_asset_id = u.compute_asset_id
    AND p.tag_key = u.TagKey
    AND (p.tag_value IS NULL OR p.tag_value = "" OR p.tag_value = u.TagValue)
    WHERE u.TagKey IS NULL
),

clean_tag_matches AS (
SELECT 
    tpm.*, 
    collect_set(up.UnmatchedPolicyKey) AS MissingPolicyKeys
FROM 
    tag_potential_matches tpm
LEFT JOIN 
    unmatched_policies up 
ON 
    tpm.compute_asset_id = up.compute_asset_id
GROUP BY 
    tpm.compute_asset_id,
    tpm.billing_origin_product,
    tpm.TotalPolicyTags,
    tpm.NumberOfMatchedKeys,
    tpm.NumberOfMatchedValues,
    tpm.MatchingTagKeys,
    tpm.MatchingTagValues,
    tpm.IsTaggingMatch,
    tpm.TagCombos,
    tpm.updated_tags
),

px_all AS (
  SELECT
    DISTINCT sku_name,
    pricing.default AS unit_price,
    unit_price :: decimal(10, 3) AS sku_price
  FROM
    system.billing.list_prices QUALIFY ROW_NUMBER() OVER (
      PARTITION BY sku_name
      ORDER BY
        price_start_time DESC
    ) = 1
),

final_parsed_query AS (
  SELECT
    u.*,
    -- TO DO: Add Discounts Table Later
    ((1 - COALESCE(NULL, 0)) * sku_price) * usage_quantity AS Dollar_DBUs,
    -- Combine system tags with App tags
    -- Combine system tags with App tags
    u.clean_tags AS updated_tags,
    (
      SELECT
        COUNT(0)
      FROM
        active_tags
    ) AS TotalPolicyTags,
    COALESCE(ct.MatchingTagKeys, array()) AS MatchingTagKeys,
    COALESCE(ct.MissingPolicyKeys, (SELECT collect_set(tag_key) FROM active_tags)) AS MissingTagKeys,
    COALESCE(ct.NumberOfMatchedKeys, 0) AS NumberOfMatchedKeys,
    COALESCE(ct.MatchingTagValues, array()) AS MatchedTagValues,
    COALESCE(ct.MatchingTagKeys, array()) AS MatchedTagKeys,
    COALESCE(ct.IsTaggingMatch, 'Not Matched To Tag Policy') AS IsTaggingMatch,
    ct.TagCombos AS TagCombos
  FROM
    filtered_usage AS u
    INNER JOIN px_all AS px ON px.sku_name = u.sku_name --- Join up tags persisted from the app
    LEFT JOIN clean_tag_matches ct ON (
      ct.compute_asset_id = u.clean_cluster_id
      AND u.billing_origin_product = ct.billing_origin_product
    )
    OR (
      ct.compute_asset_id = u.clean_job_or_pipeline_id
      AND u.billing_origin_product = ct.billing_origin_product
    )
    OR (
      ct.compute_asset_id = u.clean_warehouse_id
      AND u.billing_origin_product = ct.billing_origin_product
    )
),
filtered_result AS (
  -- Final Query - Dynamic from Filters
  SELECT
    *
  FROM
    final_parsed_query AS f
    WHERE 1=1