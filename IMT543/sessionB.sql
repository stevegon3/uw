BEGIN;
-- this UPDATE needs the same row lock and will wait
UPDATE inventory SET qty = qty - 1 WHERE id = 1;

COMMIT;