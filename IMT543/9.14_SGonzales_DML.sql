-- Steve Gonzales IMT453
-- 9.14 Project Deliverable 3 - DML Statements
-- Insert statements for all tables
-- SWE 1.5 LLM used to create sample data after prompted with a single INSERT statement for each table.

-- Insert into institution (parent table for accounts)
INSERT INTO institution (institution_name) VALUES
('Bank of America'),
('Chase'),
('Wells Fargo'),
('Credit Union'),
('Vanguard');

-- Insert into category
INSERT INTO category (category, cat_sub, cat_super) VALUES
('Groceries', 'Food', 'Essential'),
('Rent', 'Housing', 'Essential'),
('Utilities', 'Housing', 'Essential'),
('Gasoline', 'Transportation', 'Essential'),
('Entertainment', 'Lifestyle', 'Discretionary'),
('Dining Out', 'Food', 'Discretionary'),
('Salary', 'Income', 'Primary'),
('Freelance', 'Income', 'Secondary'),
('Investment', 'Income', 'Passive');

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

-- Insert into account (requires institution_id)
INSERT INTO account (account_name, institution_id, account_number) VALUES
('Checking Account', 1, '123456789'),
('Savings Account', 1, '987654321'),
('Credit Card', 2, '456789123'),
('Investment Account', 5, '789123456'),
('Joint Checking', 3, '321654987');

-- Insert into financial_transaction (supertype - requires user_id)
INSERT INTO financial_transaction (user_id, amount, description, transaction_date, transaction_type) VALUES
(1, 3500.00, 'Monthly Salary', '2024-01-15', 'INCOME'),
(1, 150.00, 'Grocery Shopping', '2024-01-16', 'EXPENSE'),
(1, 1200.00, 'Rent Payment', '2024-01-01', 'EXPENSE'),
(1, 500.00, 'Freelance Project', '2024-01-20', 'INCOME'),
(2, 2800.00, 'Bi-weekly Salary', '2024-01-15', 'INCOME'),
(2, 85.50, 'Gas Purchase', '2024-01-17', 'EXPENSE'),
(3, 4500.00, 'Monthly Salary', '2024-01-15', 'INCOME'),
(3, 200.00, 'Entertainment', '2024-01-18', 'EXPENSE');

-- Insert into expense_details (subtype - requires transaction_id, category_id, account_id, merchant_id)
INSERT INTO expense_details (transaction_id, category, payment_method, category_id, account_id, merchant_id, is_recurring, receipt_url) VALUES
(2, 'Groceries', 'Debit Card', 1, 1, 1, false, 'https://receipts.example.com/groceries123'),
(3, 'Rent', 'Bank Transfer', 2, 1, NULL, true, NULL),
(6, 'Gasoline', 'Credit Card', 4, 3, 3, false, 'https://receipts.example.com/gas456'),
(8, 'Entertainment', 'Credit Card', 5, 3, 2, false, 'https://receipts.example.com/ent789');

-- Insert into income_details (subtype - requires transaction_id, account_id)
INSERT INTO income_details (transaction_id, account_id, source, income_type, is_taxable, employer_name, payment_frequency, gross_amount, net_amount) VALUES
(1, 2, 'Tech Corp', 'Salary', true, 'Tech Corp', 'Monthly', 4500.00, 3500.00),
(4, 1, 'Web Design Project', 'Freelance', true, 'Self Employed', 'One Time', 650.00, 500.00),
(5, 2, 'Healthcare Inc', 'Salary', true, 'Healthcare Inc', 'Bi-weekly', 3800.00, 2800.00),
(7, 2, 'Finance Co', 'Salary', true, 'Finance Co', 'Monthly', 6000.00, 4500.00);

-- Insert into budget (requires user_id)
INSERT INTO budget (user_id, category_name, monthly_limit, current_spent, budget_year, budget_month) VALUES
(1, 'Groceries', 500.00, 150.00, 2024, 1),
(1, 'Rent', 1500.00, 1200.00, 2024, 1),
(1, 'Entertainment', 200.00, 50.00, 2024, 1),
(2, 'Gasoline', 150.00, 85.50, 2024, 1),
(2, 'Dining Out', 300.00, 0.00, 2024, 1),
(3, 'Entertainment', 400.00, 200.00, 2024, 1);

-- Insert into savings_goal (requires user_id)
INSERT INTO savings_goal (user_id, goal_name, target_amount, current_amount, target_date, priority_level, is_active) VALUES
(1, 'Emergency Fund', 10000.00, 2500.00, '2024-12-31', 1, true),
(1, 'Vacation Fund', 3000.00, 500.00, '2024-06-30', 3, true),
(2, 'House Down Payment', 50000.00, 12000.00, '2025-06-30', 2, true),
(3, 'Retirement Fund', 100000.00, 45000.00, '2030-12-31', 1, true),
(3, 'New Car', 25000.00, 2000.00, '2024-08-31', 4, true);