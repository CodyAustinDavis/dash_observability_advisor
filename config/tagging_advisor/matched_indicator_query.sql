-- MATCHED INDICATOR QUERY
SELECT 
SUM(CASE WHEN IsTaggingMatch = 'In Policy' THEN Dollar_DBUs_List ELSE 0 END) AS `Matched Usage Amount`,
SUM(CASE WHEN IsTaggingMatch = 'Not Matched To Tag Policy' THEN Dollar_DBUs_List ELSE 0 END) AS `Not Matched Usage Amount`
FROM filtered_result