
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

final_parsed_query AS (
  SELECT
    *,
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
            map_keys(u.custom_tags),
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
              MAP_KEYS(custom_tags),
              key -> CONCAT(key, '=', custom_tags [key])
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
      map_keys(u.custom_tags)
    ) AS MissingTagKeys,
    size(MatchingTagKeys) AS NumberOfMatchedKeys,
    array_join(MatchingTagKeys, '_') AS TagPolicyKeys,
    array_join(
      CONCAT(
        transform(
          MatchingTagKeys,
          key -> CONCAT(key, '=', u.custom_tags [key])
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
    clean_usage AS u
)