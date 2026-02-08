-- Sample INSERT statements for the first 5 tables in 5.14_SGonzales.sql

-- 1. institution table sample data
INSERT INTO money.institution (institution_name) VALUES
('Chase Bank'),
('Bank of America'),
('Wells Fargo'),
('US Bank'),
('Credit Union of Washington'),
('Ally Bank'),
('Capital One'),
('PNC Bank');

-- 2. account table sample data
INSERT INTO money.account (account_name, institution_id, account_number) VALUES
('John Checking Account', 1, '123456789'),
('John Savings Account', 1, '987654321'),
('Joint Checking', 2, '456789123'),
('Emergency Fund', 3, '789123456'),
('Vacation Savings', 4, '321654987'),
('Business Account', 5, '654987321'),
('Investment Account', 6, '147258369'),
('Credit Card', 7, '369258147');

-- 3. category table sample data
INSERT INTO money.category (category, cat_sub, cat_super) VALUES
('Groceries', 'Food', 'Essential'),
('Restaurants', 'Food', 'Non-essential'),
('Gas', 'Transportation', 'Essential'),
('Public Transit', 'Transportation', 'Essential'),
('Car Insurance', 'Transportation', 'Essential'),
('Movies', 'Entertainment', 'Non-essential'),
('Streaming Services', 'Entertainment', 'Non-essential'),
('Gym Membership', 'Health', 'Essential'),
('Doctor Visits', 'Health', 'Essential'),
('Rent', 'Housing', 'Essential'),
('Utilities', 'Housing', 'Essential'),
('Clothing', 'Personal', 'Non-essential'),
('Books', 'Education', 'Non-essential'),
('Online Courses', 'Education', 'Non-essential'),
('Savings Transfer', 'Savings', 'Essential');

-- 4. rule table sample data
INSERT INTO money.rule (rule_name, payee, payee_new, cat_new) VALUES
('Amazon Prime', 'AMZN Prime', 'Amazon', 'Entertainment'),
('Netflix', 'NETFLIX.COM', 'Netflix', 'Entertainment'),
('Starbucks', 'STARBUCKS', 'Starbucks', 'Restaurants'),
('Gas Station', 'SHELL', 'Shell Gas', 'Transportation'),
('Grocery Store', 'WHOLE FOODS', 'Whole Foods', 'Groceries'),
('Utility Bill', 'PSE', 'Puget Sound Energy', 'Utilities'),
('Phone Bill', 'VERIZON', 'Verizon', 'Utilities'),
('Insurance', 'GEICO', 'Geico', 'Transportation'),
('Gym', 'PLANET FITNESS', 'Planet Fitness', 'Health'),
('Subscription', 'ADOB*', 'Adobe', 'Software');

-- 5. merchant table sample data
INSERT INTO money.merchant (merchant_name, merchant_type) VALUES
('Amazon', 'Online Retailer'),
('Walmart', 'Retail Store'),
('Target', 'Retail Store'),
('Costco', 'Warehouse Club'),
('Safeway', 'Grocery Store'),
('Whole Foods', 'Grocery Store'),
('Shell Gas Station', 'Gas Station'),
('Chevron', 'Gas Station'),
('Starbucks', 'Coffee Shop'),
('McDonalds', 'Fast Food'),
('Netflix', 'Streaming Service'),
('Spotify', 'Music Service'),
('Apple Store', 'Electronics'),
('Best Buy', 'Electronics'),
('Home Depot', 'Home Improvement'),
('Lowes', 'Home Improvement'),
('REI', 'Outdoor Equipment'),
('Nike', 'Clothing'),
('Gap', 'Clothing'),
('CVS Pharmacy', 'Pharmacy'),
('Walgreens', 'Pharmacy');
