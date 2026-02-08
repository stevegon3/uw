-- Money Management Database - Alternative Schema
-- Five tables for personal finance tracking

--1. Users table - tracks user accounts
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
