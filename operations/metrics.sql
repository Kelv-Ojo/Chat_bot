-- Calendar elapsed hours at :asof. Inputs use canonical ISO timestamps.
WITH snapshot AS (
 SELECT *,
  first_reply_at IS NOT NULL AND julianday(first_reply_at)<=julianday(:asof) AS answered,
  resolved_at IS NOT NULL AND julianday(resolved_at)<=julianday(:asof) AS resolved
 FROM enquiries WHERE julianday(opened_at)<=julianday(:asof)
)
SELECT COUNT(*),COALESCE(SUM(NOT resolved),0),COALESCE(SUM(NOT answered),0),
 AVG(CASE WHEN answered THEN (julianday(first_reply_at)-julianday(opened_at))*24 END),
 AVG(CASE WHEN resolved THEN (julianday(resolved_at)-julianday(opened_at))*24 END),
 COALESCE(SUM(NOT resolved AND (julianday(:asof)-julianday(opened_at))*24>48),0)
FROM snapshot;
