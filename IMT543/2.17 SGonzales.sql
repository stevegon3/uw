--Assignment 2.17 IMT543
--Steve Gonzales  Postgres is allowed as per Professor Gridiron

--Question 1: Write a query to retrieve all records from the authors table.
SELECT * FROM authors;

--Question 2: Write a query to fetch all records from the titles table.
SELECT * FROM titles;

--Question 3: Create a query that displays all records from the publishers table, returning all columns for each publisher.
SELECT * FROM publishers;

--Question 4: Query the sales table to fetch all records, including all fields like store, date, and sales amount.
SELECT stor_id store, ord_date order_date, qty, ytd_sales FROM sales s JOIN titles t ON s.title_id=t.title_id;

--Question 5: Write a query to retrieve every row from the stores table. This should display information such as store name, city, and state.
SELECT * FROM stores;

--Question 6: Write a query to select all authors from the authors table whose state is 'CA' (California).
SELECT * FROM authors WHERE state = 'CA';

--Question 7: Write a query to retrieve all titles from the titles table where the price is greater than $20.
SELECT * FROM titles WHERE price>20;

--Question 8: Write a query to retrieve authors who are based in either New York (NY) or California (CA).
SELECT * FROM authors WHERE state IN ('CA', 'NY');

--Question 9: Write a query to fetch titles that are either in the 'business' category or priced below $15.
SELECT * FROM titles WHERE TYPE IN ('business') OR price < 15;

--Question 10: Create a sample table with three or four columns and a primary key.
DROP TABLE IF EXISTS sample_table;
CREATE TABLE sample_table(
  id int NOT NULL PRIMARY KEY,
  sample_varchar varchar(100),
  sample_timestamp timestamp,
  sample_real real
  );
INSERT INTO sample_table(id, sample_varchar, sample_timestamp, sample_real) VALUES (1, 'test', '2025-01-01'::timestamp, 123.12);
SELECT * FROM sample_table;

--Output of queries:
[2026-01-10 19:48:26] Connected to pubs
[2026-01-10 19:48:26] completed in 97 ms
[2026-01-10 19:48:34] uw> set search_path = "imt543"
[2026-01-10 19:48:34] completed in 5 ms
[2026-01-10 19:51:22] uw.imt543> SELECT * FROM authors
[2026-01-10 19:51:22] 23 rows retrieved starting from 1 in 78 ms (execution: 8 ms, fetching: 70 ms)
[2026-01-10 19:51:22] uw.imt543> SELECT * FROM titles
[2026-01-10 19:51:22] 18 rows retrieved starting from 1 in 100 ms (execution: 13 ms, fetching: 87 ms)
[2026-01-10 19:51:22] uw.imt543> SELECT * FROM publishers
[2026-01-10 19:51:22] 8 rows retrieved starting from 1 in 78 ms (execution: 7 ms, fetching: 71 ms)
[2026-01-10 19:51:22] uw.imt543> SELECT stor_id store, ord_date order_date, qty, ytd_sales FROM sales s JOIN titles t ON s.title_id=t.title_id
[2026-01-10 19:51:22] 21 rows retrieved starting from 1 in 66 ms (execution: 8 ms, fetching: 58 ms)
[2026-01-10 19:51:22] uw.imt543> SELECT * FROM stores
[2026-01-10 19:51:22] 6 rows retrieved starting from 1 in 77 ms (execution: 17 ms, fetching: 60 ms)
[2026-01-10 19:51:22] uw.imt543> SELECT * FROM authors WHERE state = 'CA'
[2026-01-10 19:51:23] 15 rows retrieved starting from 1 in 52 ms (execution: 6 ms, fetching: 46 ms)
[2026-01-10 19:51:23] uw.imt543> SELECT * FROM titles WHERE price>20
[2026-01-10 19:51:23] 3 rows retrieved starting from 1 in 60 ms (execution: 7 ms, fetching: 53 ms)
[2026-01-10 19:51:23] uw.imt543> SELECT * FROM authors WHERE state IN ('CA', 'NY')
[2026-01-10 19:51:23] 15 rows retrieved starting from 1 in 46 ms (execution: 6 ms, fetching: 40 ms)
[2026-01-10 19:51:23] uw.imt543> SELECT * FROM titles WHERE TYPE IN ('business') OR price < 15
[2026-01-10 19:51:23] 10 rows retrieved starting from 1 in 49 ms (execution: 7 ms, fetching: 42 ms)
[2026-01-10 19:51:23] uw.imt543> DROP TABLE IF EXISTS sample_table
[2026-01-10 19:51:23] completed in 9 ms
[2026-01-10 19:51:23] uw.imt543> CREATE TABLE sample_table(
                                   id int NOT NULL PRIMARY KEY,
                                   sample_varchar varchar(100),
                                   sample_timestamp timestamp,
                                   sample_real real
                                   )
[2026-01-10 19:51:23] completed in 18 ms
[2026-01-10 19:51:23] uw.imt543> INSERT INTO sample_table(id, sample_varchar, sample_timestamp, sample_real) VALUES (1, 'test', '2025-01-01'::timestamp, 123.12)
[2026-01-10 19:51:23] 1 row affected in 8 ms
[2026-01-10 19:51:23] uw.imt543> SELECT * FROM sample_table
[2026-01-10 19:51:23] 1 row retrieved starting from 1 in 61 ms (execution: 5 ms, fetching: 56 ms)
[2026-01-10 19:51:23] uw.imt543> SELECT 'authors' tbl, count(*) ct FROM authors UNION
                                 SELECT 'titles' tbl, count(*) ct FROM titles UNION
                                 SELECT 'titeauthor' tbl, count(*) ct FROM titleauthor
[2026-01-10 19:51:23] 3 rows retrieved starting from 1 in 46 ms (execution: 8 ms, fetching: 38 ms)

SELECT 'authors' tbl, count(*) ct FROM authors UNION
SELECT 'titles' tbl, count(*) ct FROM titles UNION
SELECT 'titeauthor' tbl, count(*) ct FROM titleauthor;