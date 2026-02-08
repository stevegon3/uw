/* Five new tables: You need to create at least five tables in total.
Primary Keys: One or two tables should have a primary key.
Foreign Key: One table should have a foreign key that references another table.
Submit:
SQL Queries used to create the tables, insert data, and perform other operations.
Screenshots of the output from the SQL console.
*/

--1. Five new tables
CREATE SCHEMA IF NOT EXISTS money;

DROP TABLE IF EXISTS money.institution CASCADE;
CREATE TABLE money.institution (
  institution_id  serial PRIMARY KEY,
  institution_name  VARCHAR(30));

DROP TABLE IF EXISTS money.account CASCADE;
CREATE TABLE money.account (
  account_id  serial PRIMARY KEY,
  account_name  VARCHAR(30),
  institution_id int REFERENCES money.institution(institution_id),
  account_number varchar(30));

DROP TABLE IF EXISTS money.category CASCADE;
CREATE TABLE money.category (
    category_id         SERIAL PRIMARY KEY,
    category            VARCHAR(100),
    cat_sub             VARCHAR(25),
    cat_super           VARCHAR(25));

DROP TABLE IF EXISTS money.rule CASCADE;
CREATE TABLE money.rule (
    rule_id SERIAL PRIMARY KEY,
    rule_name varchar(50),
    payee varchar(150),
    payee_new varchar(50) ,
    cat_new varchar(100));

DROP TABLE IF EXISTS money.merchant CASCADE;
CREATE TABLE money.merchant (
  merchant_id  serial PRIMARY KEY,
  merchant_name  VARCHAR(30),
  merchant_type varchar(30));

DROP TABLE IF EXISTS money.transaction CASCADE;
CREATE TABLE IF NOT EXISTS money.transaction (
    transaction_id      SERIAL PRIMARY KEY,
    merchant            VARCHAR(100),
    transaction_dt      DATE,
    category_id         int REFERENCES money.category(category_id),
    amt                 FLOAT,
    account_id          int REFERENCES money.account(account_id),
    merchant_id         int REFERENCES money.merchant(merchant_id),
    rule_id             int REFERENCES money.rule(rule_id));

DROP TABLE IF EXISTS money.users CASCADE;
CREATE TABLE money.users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

--2. Budget table - tracks budget categories and limits
DROP TABLE IF EXISTS money.budget CASCADE;
CREATE TABLE money.budget (
    budget_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES money.users(user_id) ON DELETE CASCADE,
    category_name VARCHAR(100) NOT NULL,
    monthly_limit DECIMAL(10,2) NOT NULL,
    current_spent DECIMAL(10,2) DEFAULT 0.00,
    budget_year INT NOT NULL,
    budget_month INT NOT NULL,
    UNIQUE(user_id, category_name, budget_year, budget_month)
);

--3. Expense table - tracks individual expenses
DROP TABLE IF EXISTS money.expense CASCADE;
CREATE TABLE money.expense (
    expense_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES money.users(user_id) ON DELETE CASCADE,
    amount DECIMAL(10,2) NOT NULL,
    description TEXT,
    expense_date DATE NOT NULL,
    category VARCHAR(100) NOT NULL,
    payment_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

--4. Income table - tracks income sources
DROP TABLE IF EXISTS money.income CASCADE;
CREATE TABLE money.income (
    income_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES money.users(user_id) ON DELETE CASCADE,
    amount DECIMAL(10,2) NOT NULL,
    source VARCHAR(100) NOT NULL,
    income_date DATE NOT NULL,
    income_type VARCHAR(50) DEFAULT 'salary',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

--5. Savings_goal table - tracks savings objectives
DROP TABLE IF EXISTS money.savings_goal CASCADE;
CREATE TABLE money.savings_goal (
    goal_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES money.users(user_id) ON DELETE CASCADE,
    goal_name VARCHAR(100) NOT NULL,
    target_amount DECIMAL(10,2) NOT NULL,
    current_amount DECIMAL(10,2) DEFAULT 0.00,
    target_date DATE,
    priority_level INT DEFAULT 1 CHECK (priority_level BETWEEN 1 AND 5),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add some sample data
-- 1. institution table sample data
INSERT INTO money.institution (institution_id, institution_name) VALUES
(1,'Chase Bank'),
(2, 'Bank of America'),
(3, 'Wells Fargo'),
(4, 'US Bank'),
(5, 'Credit Union of Washington'),
(6, 'Ally Bank'),
(7, 'Capital One'),
(8, 'PNC Bank');

-- 2. account table sample data
select count(*) from money.institution WHERE institution_id=1;

INSERT INTO money.account (account_name, institution_id, account_number) VALUES
('Jon Checking Account', 1, '123456789'),
('Jon Savings Account', 1, '987654321'),
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
('Streaming', 'Entertainment', 'Entertainment');

-- 4. rule table sample data
INSERT INTO money.rule (rule_name, payee, payee_new, cat_new) VALUES
('Netflix', 'NETFLIX.COM', 'Netflix', 'Entertainment:Streaming'),
('Starbucks', 'STARBUCKS', 'Starbucks', 'Entertainment:Restaurants'),
('Subscription', 'ADOB*', 'Adobe', 'Subscription:Software');

-- 5. merchant table sample data
INSERT INTO money.merchant (merchant_name, merchant_type) VALUES
('Walmart', 'Retail Store'),
('Target', 'Retail Store'),
('Costco', 'Warehouse Club'),
('Safeway', 'Grocery Store');

INSERT INTO money.users (username, email, first_name, last_name) VALUES
('john_doe', 'john@example.com', 'John', 'Doe'),
('jane_smith', 'jane@example.com', 'Jane', 'Smith');

INSERT INTO money.budget (user_id, category_name, monthly_limit, budget_year, budget_month) VALUES
(1, 'Groceries', 500.00, 2024, 1),
(1, 'Transportation', 200.00, 2024, 1),
(1, 'Entertainment', 150.00, 2024, 1);

INSERT INTO money.expense (user_id, amount, description, expense_date, category, payment_method) VALUES
(1, 45.67, 'Grocery shopping at Whole Foods', '2024-01-15', 'Groceries', 'Credit Card'),
(1, 25.00, 'Gas station', '2024-01-16', 'Transportation', 'Debit Card'),
(1, 35.50, 'Movie tickets', '2024-01-17', 'Entertainment', 'Credit Card');

INSERT INTO money.income (user_id, amount, source, income_date, income_type) VALUES
(1, 3500.00, 'Monthly salary', '2024-01-01', 'salary'),
(1, 500.00, 'Freelance project', '2024-01-15', 'freelance');

INSERT INTO money.savings_goal (user_id, goal_name, target_amount, target_date, priority_level) VALUES
(1, 'Emergency Fund', 10000.00, '2024-12-31', 1),
(1, 'Vacation', 3000.00, '2024-06-30', 3),
(1, 'New Laptop', 1500.00, '2024-03-31', 2);
