, exploded_tags AS 
(SELECT
usage_date,
explode(TagCombos) AS PolicyTagValue,
Dollar_DBUs_List AS Dollar_DBUs_List
FROM 
filtered_result
)
-- TAG VALUES QUERY OVER TIME LINE CHART
SELECT usage_date AS `Usage Date`,
CASE WHEN len(PolicyTagValue) = 0 
    THEN 'Not In Policy' ELSE PolicyTagValue END 
    AS `Tag Value In Policy`,
SUM(Dollar_DBUs_List) AS `Usage Amount`
FROM exploded_tags
GROUP BY usage_date, CASE WHEN len(PolicyTagValue) = 0 THEN 'Not In Policy' ELSE PolicyTagValue END