WITH parsed_tagging_table AS (
  SELECT
    tag_key AS key,
    COALESCE(tag_value, '') AS value,
    CONCAT(
      COALESCE(tag_key, ''),
      CASE
        WHEN len(tag_value) > 0 THEN '='
        ELSE ''
      END,
      COALESCE(tag_value)
    ) AS combination,
    tag_policy_name,
    CASE
      WHEN len(tag_value) > 0 THEN '='
      ELSE 0
    END AS ContainsValuePair
  FROM
    app_tag_policies
  WHERE
    1 = 1
