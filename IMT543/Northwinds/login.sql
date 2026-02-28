CREATE ROLE nw_login_1 WITH LOGIN PASSWORD 'imt543';
CREATE ROLE nw_login_2 WITH LOGIN PASSWORD 'imt543';
CREATE ROLE nw_login_3 WITH LOGIN PASSWORD 'imt543';
CREATE ROLE nw_login_4 WITH LOGIN PASSWORD 'imt543';
SELECT * FROM pg_roles WHERE rolname iLIKE 'nw_%' ORDER BY rolname;

-- Table permissions assignment
-- Table 1 (First login): Select, insert, update
GRANT SELECT, INSERT, UPDATE ON Employees TO nw_login_1;

-- Table 2 (Second Login): Select
GRANT SELECT ON Categories TO nw_login_2;

-- Table 3 (Third Login): Insert, Update, Delete
GRANT INSERT, UPDATE, DELETE ON Customers TO nw_login_3;

-- Table 4 (Fourth Login): Select, Insert
GRANT SELECT, INSERT ON Shippers TO nw_login_4;

SET ROLE postgres;
SELECT 'current_user' lbl, current_user user_name UNION
SELECT 'session_user' lbl, session_user user_name;
INSERT INTO customers (customerid, companyname, contactname, contacttitle, address, city, region, postalcode, country, phone, fax)
VALUES('ALFKI','Alfreds Futterkiste','Maria Anders','Sales Representative','Obere Str. 57','Berlin',NULL,'12209','Germany','030-0074321','030-0076545');

SET ROLE nw_login_2;
SELECT 'current_user' lbl, current_user user_name UNION
SELECT 'session_user' lbl, session_user user_name;

SET ROLE postgres;
GRANT USAGE ON SCHEMA northwinds TO nw_login_2;
GRANT SELECT ON northwinds.customers TO nw_login_2;

SET ROLE nw_login_2;
SELECT * FROM northwinds.customers;
INSERT INTO customers (customerid, companyname, contactname, contacttitle, address, city, region, postalcode, country, phone, fax)
VALUES('ANTON','Antonio Moreno Taquería','Antonio Moreno','Owner','Mataderos  2312','México D.F.',NULL,'05023','Mexico','(5) 555-3932',NULL);
