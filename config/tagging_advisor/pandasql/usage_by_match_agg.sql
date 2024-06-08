SELECT usage_date AS Usage_Date, 
SUM(Dollar_DBUs_List) AS `Usage Amount`,
IsTaggingMatch AS `Tag Match`
FROM df
GROUP BY usage_date, IsTaggingMatch