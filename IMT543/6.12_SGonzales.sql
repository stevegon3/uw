/*
Write a query to find all authors who have no middle name (assuming the Authors table has a middle_name column). Use the IS NULL operator to filter for NULL values.
Write a query that extracts the first three characters of the title from the Titles table (assuming there’s a title column).
Write a query that removes leading spaces from the publisher_name in the Publishers table.
Write a query that removes trailing spaces from the author_name in the Authors table.
Write a query to find all book titles that start with the letter "T" (assuming there’s a title column in the Titles table). Use the LIKE operator and a wildcard to match the pattern.
Submission Requirements:
Execute each query in the SQL database.
Provide screenshots of the output from each query.
Submit both the SQL queries and output screenshots in a single Word or PDF file.
*/

--1. all authors who have no middle name
SELECT * FROM authors;
--Middle name does not exist, so let's add it.
--First "backup the table"
ALTER TABLE authors ADD COLUMN au_mname varchar(50);
--Put some random data in it
UPDATE authors SET au_mname=substr(city,1,4) || substr(au_fname,1,2) WHERE state='CA';
SELECT * FROM authors WHERE au_mname IS NULL;
ALTER TABLE authors DROP COLUMN au_mname;

--2. First 3 of title
SELECT substr(title,1,3) FROM titles;

--3. Left Strip publisher_name
SELECT pub_name,ltrim(pub_name),* FROM publishers;
--None have leading spaces.
INSERT INTO publishers (pub_id, pub_name, city, state, country) VALUES ('9988', ' test name with leading spaces', 'Oxford', 'UK', 'USA');
SELECT pub_name,ltrim(pub_name),* FROM publishers WHERE pub_name != ltrim(pub_name);
DELETE FROM publishers WHERE pub_id='9988';

--4. Right Strip author_name
SELECT au_lname,rtrim(au_lname),* FROM authors;
--None have trailing spaces, so insert one
INSERT INTO imt543.authors
(au_id, au_lname, au_fname, phone, address, city, state, zip, contract)
VALUES('9988', 'test trail space ', 'test trail space ', '206-555-1212', '', '', '', '98125', false);
SELECT au_lname,length(au_lname),length(rtrim(au_lname)),* FROM authors WHERE length(au_lname) != length(rtrim(au_lname));
DELETE FROM authors WHERE au_id='9988';

--5. Find all titles that start with T
SELECT * FROM titles WHERE title LIKE 'T%';