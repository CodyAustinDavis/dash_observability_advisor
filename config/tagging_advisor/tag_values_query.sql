, exploded_tags AS 
(SELECT
explode(TagCombos) AS PolicyTagValue,
Dollar_DBUs_List AS Dollar_DBUs_List
FROM 
filtered_result
)
-- TAG VALUES QUERY HISTOGRAM
SELECT CASE WHEN len(PolicyTagValue) = 0 
    THEN 'Not In Policy' ELSE PolicyTagValue END 
    AS `Tag Value In Policy`,
SUM(Dollar_DBUs_List) AS `Usage Amount`
FROM exploded_tags
GROUP BY CASE WHEN len(PolicyTagValue) = 0 THEN 'Not In Policy' ELSE PolicyTagValue END 
ORDER BY `Usage Amount` DESC