, exploded_tags AS (
SELECT
    explode(TagCombos) AS PolicyTagValue,
    Dollar_DBUs_List AS Usage_Amount,
    billing_origin_product AS Product
FROM
    filtered_result
)
-- TAG/SKU HEATMAP QUERY
SELECT 
PolicyTagValue AS Tag,
Product AS Product,
SUM(Usage_Amount) AS `Usage Amount`
FROM exploded_tags
GROUP BY  PolicyTagValue,
Product