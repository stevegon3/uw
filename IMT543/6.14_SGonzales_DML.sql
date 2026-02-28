-- Steve Gonzales IMT453
-- 6.14 Project Deliverable 3 - DML Statements
-- Insert statements for all tables
-- SWE 1.5 LLM used to create sample data after prompted with a single INSERT statement for each table.

-- Insert into institution (parent table for accounts)
INSERT INTO institution (institution_name) VALUES 
('Bank of America'),
('Chase'),
('Wells Fargo'),
('Credit Union'),
('Vanguard');

-- Insert into merchant
INSERT INTO merchant (merchant_name, merchant_type) VALUES 
('Safeway', 'Grocery'),
('Target', 'Retail'),
('Shell', 'Gas Station'),
('Netflix', 'Subscription'),
('Starbucks', 'Restaurant'),
('Amazon', 'E-commerce'),
('McDonalds', 'Fast Food'),
('Home Depot', 'Hardware');

-- Insert into users
INSERT INTO users (username, email, first_name, last_name, address, city, zip_code, zip_plus_4) VALUES 
('sgonzales', 'steve.gonzales@email.com', 'Steve', 'Gonzales', '123 Main St', 'Seattle', '98101', '1234'),
('jdoe', 'jane.doe@email.com', 'Jane', 'Doe', '456 Oak Ave', 'Bellevue', '98004', '5678'),
('bsmith', 'bob.smith@email.com', 'Bob', 'Smith', '789 Pine Rd', 'Redmond', '98052', '9012');

/*
 [2026-02-14 20:10:13] uw.money> INSERT INTO institution (institution_name) VALUES
                                ('Bank of America'),
                                ('Chase'),
                                ('Wells Fargo'),
                                ('Credit Union'),
                                ('Vanguard')
[2026-02-14 20:10:13] 5 rows affected in 8 ms
[2026-02-14 20:10:13] uw.money> INSERT INTO merchant (merchant_name, merchant_type) VALUES
                                ('Safeway', 'Grocery'),
                                ('Target', 'Retail'),
                                ('Shell', 'Gas Station'),
                                ('Netflix', 'Subscription'),
                                ('Starbucks', 'Restaurant'),
                                ('Amazon', 'E-commerce'),
                                ('McDonalds', 'Fast Food'),
                                ('Home Depot', 'Hardware')
[2026-02-14 20:10:13] 8 rows affected in 9 ms
[2026-02-14 20:10:13] uw.money> INSERT INTO users (username, email, first_name, last_name, address, city, zip_code, zip_plus_4) VALUES
                                ('sgonzales', 'steve.gonzales@email.com', 'Steve', 'Gonzales', '123 Main St', 'Seattle', '98101', '1234'),
                                ('jdoe', 'jane.doe@email.com', 'Jane', 'Doe', '456 Oak Ave', 'Bellevue', '98004', '5678'),
                                ('bsmith', 'bob.smith@email.com', 'Bob', 'Smith', '789 Pine Rd', 'Redmond', '98052', '9012')
[2026-02-14 20:10:13] 3 rows affected in 7 ms

 */