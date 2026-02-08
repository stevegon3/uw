--9. Write an ALTER TABLE query to add a new column named author (VARCHAR(100)) to the Books table.
SELECT column_name,data_type
FROM information_schema.columns WHERE table_schema iLIKE 'imt543' AND table_name ilike 'Books' ORDER BY table_name, column_name;
ALTER TABLE Books ADD COLUMN author varchar(100);
SELECT column_name,data_type
FROM information_schema.columns WHERE table_schema iLIKE 'imt543' AND table_name ilike 'Books' ORDER BY table_name, column_name;

--10. Write a SQL query to drop the Books table from the database.
DROP TABLE Books;
SELECT table_name FROM information_schema.tables WHERE table_schema = 'imt543' ORDER BY table_name;