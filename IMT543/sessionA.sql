DROP TABLE IF EXISTS inventory;
CREATE TABLE inventory (id serial PRIMARY KEY, qty int);
INSERT INTO inventory (qty) VALUES (10);

BEGIN;
-- acquires a row lock on id=1 and holds it until COMMIT/ROLLBACK
SELECT * FROM inventory WHERE id = 1 FOR UPDATE;