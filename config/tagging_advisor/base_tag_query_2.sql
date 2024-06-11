
),

parsed_keys_only AS (
        SELECT
        collect_list(key) AS KeyOnlyPolicies
      FROM
        parsed_tagging_table
        WHERE ContainsValuePair = 0 -- Just Key Match Requirement
),


parsed_combo_policies AS (
        SELECT
        collect_list(combination) AS CombinationPolicies
      FROM
        parsed_tagging_table
        WHERE ContainsValuePair = 1 -- Both Key Value Match Requirements
),

parsed_keys_all AS (
        SELECT
        collect_list(key) AS PolicyKeys
      FROM
        parsed_tagging_table
),


px_all AS (
  SELECT DISTINCT
  sku_name,
  pricing.default AS unit_price,
  unit_price::decimal(10,3) AS sku_price
  FROM system.billing.list_prices 
  QUALIFY ROW_NUMBER() OVER (PARTITION BY sku_name ORDER BY price_start_time DESC) = 1
  ),


final_parsed_query AS (
  SELECT
    *,
    -- TO DO: Add Discounts Table Later
    ((1-COALESCE(NULL, 0))*sku_price)*usage_quantity AS Dollar_DBUs,
    -- Combine system tags with App tags
    map_zip_with(
      u.clean_tags,
      COALESCE(ct.user_tags, map()),(k, v1, v2) -> CASE WHEN length(v2) > 1 THEN v2 ELSE v1 END
    ) AS updated_tags,
    -- Get Total Keys Required
    -- Now tag combos can be matched in 2 separate ways: key only, or the key=value pair if optionall provided
    (
      SELECT
        COUNT(0) -- Number of selected Policies
      FROM
        parsed_tagging_table
    ) AS TotalPolicyTags,
    -- When only keys are provided, link keys, but when result_map has a value, check the set of the whole k=v pair
    (
      ARRAY_DISTINCT(
        CONCAT(
          --- Key Only Intersection
          array_intersect(
            map_keys(updated_tags),
            (
              SELECT
                MAX(KeyOnlyPolicies)
              FROM
                parsed_keys_only
            )
          ),

          -- Key + Value Intersection
          array_intersect(
            TRANSFORM(
              MAP_KEYS(updated_tags),
              key -> CONCAT(key, '=', updated_tags [key])
            ),
            (
              SELECT
                MAX(CombinationPolicies)
              FROM
                parsed_combo_policies
            )
          )
        )
      )
    ) AS MatchingTagKeys,

    array_except(
      (
             (
              SELECT
                MAX(PolicyKeys)
              FROM
                parsed_keys_all
            )
      ),
      map_keys(updated_tags)
    ) AS MissingTagKeys,
    size(MatchingTagKeys) AS NumberOfMatchedKeys,
    array_join(MatchingTagKeys, '_') AS TagPolicyKeys,
    array_join(
      CONCAT(
        transform(
          MatchingTagKeys,
          key -> CONCAT(key, '=', updated_tags [key])
        ),
        -- Get Compliant keys without value pair
        FILTER(MatchingTagKeys, x -> POSITION('=' IN x) > 0) -- Pull out the compliant values with the key pair
      ),
      ';'
    ) AS MatchedTagValues,
    CASE
      WHEN NumberOfMatchedKeys >= TotalPolicyTags THEN 'In Policy'
      ELSE 'Not Matched To Tag Policy'
    END AS IsTaggingMatch,
    array_distinct(split(MatchedTagValues, ";")) AS TagCombos
  FROM
    clean_usage_table AS u
  INNER JOIN px_all AS px ON px.sku_name = u.sku_name
  --- Join up tags persisted from the app
  LEFT JOIN (SELECT *, map(tag_key, tag_value) AS user_tags FROM app_compute_tags WHERE tag_key IS NOT NULL) ct ON 
      (ct.compute_asset_id = u.clean_cluster_id AND u.billing_origin_product = ct.compute_asset_type)
      OR 
      (ct.compute_asset_id = u.clean_job_or_pipeline_id AND u.billing_origin_product = ct.compute_asset_type)
      OR 
      (ct.compute_asset_id = u.clean_warehouse_id AND u.billing_origin_product = ct.compute_asset_type)
)