),

active_tags AS (
    SELECT *, 1 AS IsPolicyTag 
    FROM app_tag_policies
    WHERE 
    1=1