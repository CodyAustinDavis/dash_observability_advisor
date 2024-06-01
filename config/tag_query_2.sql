
),

    aggregated_data AS (
      SELECT 
        collect_list(key) as keys,
        collect_list(value) as values, 
        collect_list(CASE WHEN ContainsValuePair = 1 THEN combination END) as KVCombos
      FROM 
        clean_keys
    ),

    clean_aggs AS (
    SELECT 
      map_from_arrays(keys, values) AS result_map,
      KVCombos
    FROM 
      aggregated_data LIMIT 1
    )

    SELECT 
    -- map with only keys
    MAP_FILTER(result_map, (key, value) -> value = '') AS filtered_map,
    -- entries that need to match both key and value pairs
    KVCombos
    FROM clean_aggs
),

final_parsed_query AS (
SELECT 
*,
-- Get Total Keys Required
-- Now tag combos can be matched in 2 separate ways: key only, or the key=value pair if optionall provided
(SELECT MAX(size(map_keys(filtered_map)) + size(KVCombos)) FROM parsed_tagging_table) AS TotalPolicyTags,

-- When only keys are provided, link keys, but when result_map has a value, check the set of the whole k=v pair
(
  ARRAY_DISTINCT(
      CONCAT(
          array_intersect(map_keys(u.custom_tags), (SELECT MAX(map_keys(filtered_map)) FROM parsed_tagging_table)) 
          ,array_intersect(
              TRANSFORM(
              MAP_KEYS(custom_tags),
              key -> CONCAT(key, '=', custom_tags[key])
            ), (SELECT MAX(KVCombos) FROM parsed_tagging_table)
                          )
            )
  )
) AS MatchingTagKeys,

array_except((SELECT MAX(map_keys(filtered_map)) FROM parsed_tagging_table), map_keys(u.custom_tags)) AS MissingTagKeys,
size(MatchingTagKeys) AS NumberOfMatchedKeys,
array_join(MatchingTagKeys, '_') AS TagPolicyKeys,

array_join(
  CONCAT(transform(MatchingTagKeys, key -> CONCAT(key, '=', u.custom_tags[key])), -- Get Compliant keys without value pair
  FILTER(MatchingTagKeys, x -> POSITION('=' IN x) > 0) -- Pull out the compliant values with the key pair
  )
  , ';') AS MatchedTagValues,

CASE WHEN NumberOfMatchedKeys >= TotalPolicyTags THEN 'In Policy' ELSE 'Not Matched To Tag Policy' END AS IsTaggingMatch
FROM clean_usage AS u
)
