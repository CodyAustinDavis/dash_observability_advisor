WITH filtered_usage AS (
    SELECT * FROM clean_usage
    -- Query Filters
      WHERE