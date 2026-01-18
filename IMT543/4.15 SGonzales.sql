--1. Table Creation:
--Write a SQL query to create a table named Books with the following fields:
CREATE TABLE Books (
    book_id INT Primary Key,
    title  VARCHAR(100),
    publisher_id INT
);

--2. Insert Data:
--Write a query to insert one records into the Books table with appropriate values for book_id, title, and publisher_id.
-- (Please use the publisher ID from publisher table because it is configured as foreign key here)
INSERT INTO Books (book_id, title, publisher_id) VALUES (1, 'Grapes of Wrath', 1622);

--3. Update the title of a book with book_id = 1 in the Books table to a new title of your choice.
UPDATE Books SET title = 'The Grapes of Wrath' WHERE book_id=1;

--4. Write a query to delete the record from the Books table where the book_id is 1.
DELETE FROM Books WHERE book_id=1;

--5. Count the number of distinct authors from the author table.
SELECT count(DISTINCT au_id) FROM authors;
--Should be the same as count(*)
SELECT count(*) FROM authors;

--6. Write a query using an inner join with the employee and publishers table. You will then extract the data from both tables.
SELECT *
FROM employee e JOIN publishers p ON e.pub_id=p.pub_id;

--7. Write a query using a left join with the employee and publishers table. You will then extract all the data from the employee table.
SELECT *
FROM employee e LEFT JOIN publishers p ON e.pub_id=p.pub_id;

--8. Write a query using a right join with the employee and publishers table. You will then extract all the data from publisher table.
SELECT *
FROM employee e RIGHT JOIN publishers p ON e.pub_id=p.pub_id;

--9. Write an ALTER TABLE query to add a new column named author (VARCHAR(100)) to the Books table.
ALTER TABLE Books ADD COLUMN author varchar(100);

--10. Write a SQL query to drop the Books table from the database.
DROP TABLE Books;

