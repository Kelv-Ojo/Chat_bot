-- Calendar elapsed hours; averages exclude missing timestamps.
SELECT COUNT(*),SUM(resolved_at IS NULL),SUM(first_reply_at IS NULL),
 AVG((julianday(first_reply_at)-julianday(opened_at))*24),
 AVG((julianday(resolved_at)-julianday(opened_at))*24),
 SUM(resolved_at IS NULL AND (julianday(:asof)-julianday(opened_at))*24>48)
FROM enquiries WHERE opened_at<=:asof;
