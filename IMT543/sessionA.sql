DROP TABLE IF EXISTS inventory;
CREATE TABLE inventory (id serial PRIMARY KEY, qty int);
INSERT INTO inventory (qty) VALUES (10);

BEGIN;
-- acquires a row lock on id=1 and holds it until COMMIT/ROLLBACK
SELECT * FROM inventory WHERE id = 1 FOR UPDATE;

COMMIT; 

SELECT pid, now() - query_start AS duration, state, wait_event_type, wait_event, left(query,200) AS query
FROM pg_stat_activity
WHERE state = 'active' AND now() - query_start > interval '10 seconds'
ORDER BY duration DESC LIMIT 10;