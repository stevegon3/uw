-- Steve Gonzales IMT453
-- 6.14 Project Deliverable 3

-- Drop tables in reverse dependency order with CASCADE
DROP TABLE IF EXISTS financial_transaction CASCADE;
DROP TABLE IF EXISTS expense_details CASCADE;
DROP TABLE IF EXISTS income_details CASCADE;
DROP TABLE IF EXISTS budget CASCADE;
DROP TABLE IF EXISTS savings_goal CASCADE;
DROP TABLE IF EXISTS account CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS merchant CASCADE;
DROP TABLE IF EXISTS category CASCADE;
DROP TABLE IF EXISTS institution CASCADE;

-- money.category definition
CREATE TABLE category (
	category_id serial PRIMARY KEY,  --Same as SQL Server IDENTITY
	category varchar(100),
	cat_sub varchar(25),
	cat_super varchar(25)
);

-- money.institution definition
CREATE TABLE institution (
	institution_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
	institution_name varchar(30)
);

-- money.merchant definition
CREATE TABLE merchant (
	merchant_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
	merchant_name varchar(30),
	merchant_type varchar(30)
);

-- money.users definition
CREATE TABLE users (
	user_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
	username varchar(50) NOT NULL,
	email varchar(100) NOT NULL,  --This is an online system, so email is required, ie NOT NULL
	first_name varchar(50) NULL,  --Since it's an online system, Name, Adress, etc is not required NULL means nulls are allowed
	last_name varchar(50),  --Since it's an online system, Name, Adress, etc is not required (blank or NULL in Postgres means NULLs are allowed)
    address varchar(100),
    city varchar(50),
    zip_code char(5),
    zip_plus_4 char(4),
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT users_email_key UNIQUE (email),
	CONSTRAINT users_username_key UNIQUE (username)
);

-- money.account definition
CREATE TABLE account (
	account_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
	account_name varchar(30),
	institution_id int REFERENCES institution(institution_id) NOT NULL,
	account_number varchar(30)
);

-- money.budget definition
CREATE TABLE budget (
	budget_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
	user_id int REFERENCES users(user_id) NOT NULL,  --All columns are required for this sytem to make the budget functional
	category_name varchar(100) NOT NULL,
	monthly_limit numeric(10, 2) NOT NULL,
	current_spent numeric(10, 2) DEFAULT 0.00 NOT NULL,
	budget_year int4 NOT NULL,
	budget_month int4 NOT NULL,
	CONSTRAINT budget_user_id_category_name_budget_year_budget_month_key UNIQUE (user_id, category_name, budget_year, budget_month)
);

-- money.savings_goal definition
CREATE TABLE savings_goal (
	goal_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
	user_id int references users(user_id),
	goal_name varchar(100) NOT NULL,
	target_amount numeric(10, 2) NOT NULL,
	current_amount numeric(10, 2) DEFAULT 0.00 NOT NULL,
	target_date date,
	priority_level int4 DEFAULT 1 NOT NULL,
	is_active bool DEFAULT true NOT NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT savings_goal_priority_level_check CHECK (((priority_level >= 1) AND (priority_level <= 5)))
);

-- Supertype
CREATE TABLE financial_transaction (
    transaction_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
    user_id int REFERENCES users(user_id),
    amount numeric(10,2) NOT NULL,
    description text,
    transaction_date date NOT NULL,
    transaction_type varchar(20) CHECK (transaction_type IN ('EXPENSE', 'INCOME'))
);

-- Subtype for expenses
CREATE TABLE expense_details (
    transaction_id int PRIMARY KEY REFERENCES financial_transaction(transaction_id),
    category varchar(100),
    payment_method varchar(50),             -- How money left (cash, card, transfer)
    category_id int REFERENCES category(category_id),
    account_id int REFERENCES account(account_id),
    merchant_id int REFERENCES merchant(merchant_id),  -- Who received the money
    is_recurring boolean DEFAULT false,        -- Does this repeat monthly?
    receipt_url varchar(255)                  -- Proof of purchase
);

-- Subtype for income
CREATE TABLE income_details (
    transaction_id int PRIMARY KEY REFERENCES financial_transaction(transaction_id),
    account_id int REFERENCES account(account_id),
    source varchar(100) NOT NULL,              -- Where money came from (employer, client)
    income_type varchar(50),                    -- Salary, freelance, investment, gift
    is_taxable boolean DEFAULT true,            -- Tax implications
    employer_name varchar(100),                 -- For employment income
    payment_frequency varchar(20),              -- How often this income occurs
    gross_amount numeric(10,2),                 -- Before taxes/deductions
    net_amount numeric(10,2)                -- After taxes/deductions
);

/*
 [2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS financial_transaction CASCADE
table "financial_transaction" does not exist, skipping
[2026-02-14 20:02:00] completed in 6 ms
[2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS expense_details CASCADE
table "expense_details" does not exist, skipping
[2026-02-14 20:02:00] completed in 6 ms
[2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS income_details CASCADE
table "income_details" does not exist, skipping
[2026-02-14 20:02:00] completed in 5 ms
[2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS budget CASCADE
[2026-02-14 20:02:00] completed in 11 ms
[2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS savings_goal CASCADE
table "savings_goal" does not exist, skipping
[2026-02-14 20:02:00] completed in 4 ms
[2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS account CASCADE
[2026-02-14 20:02:00] completed in 8 ms
[2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS users CASCADE
[2026-02-14 20:02:00] completed in 8 ms
[2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS merchant CASCADE
[2026-02-14 20:02:00] completed in 7 ms
[2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS category CASCADE
[2026-02-14 20:02:00] completed in 6 ms
[2026-02-14 20:02:00] uw.money> DROP TABLE IF EXISTS institution CASCADE
[2026-02-14 20:02:00] completed in 7 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE category (
                                	category_id serial PRIMARY KEY,  --Same as SQL Server IDENTITY
                                	category varchar(100),
                                	cat_sub varchar(25),
                                	cat_super varchar(25)
                                )
[2026-02-14 20:02:00] completed in 16 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE institution (
                                	institution_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
                                	institution_name varchar(30)
                                )
[2026-02-14 20:02:00] completed in 13 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE merchant (
                                	merchant_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
                                	merchant_name varchar(30),
                                	merchant_type varchar(30)
                                )
[2026-02-14 20:02:00] completed in 12 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE users (
                                	user_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
                                	username varchar(50) NOT NULL,
                                	email varchar(100) NOT NULL,
                                	first_name varchar(50),
                                	last_name varchar(50),
                                    address varchar(100),
                                    city varchar(50),
                                    zip_code char(5),
                                    zip_plus_4 char(4),
                                	created_at timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL,
                                	CONSTRAINT users_email_key UNIQUE (email),
                                	CONSTRAINT users_username_key UNIQUE (username)
                                )
[2026-02-14 20:02:00] completed in 26 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE account (
                                	account_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
                                	account_name varchar(30),
                                	institution_id int REFERENCES institution(institution_id) NOT NULL,
                                	account_number varchar(30)
                                )
[2026-02-14 20:02:00] completed in 14 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE budget (
                                	budget_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
                                	user_id int REFERENCES users(user_id) NOT NULL,
                                	category_name varchar(100) NOT NULL,
                                	monthly_limit numeric(10, 2) NOT NULL,
                                	current_spent numeric(10, 2) DEFAULT 0.00 NOT NULL,
                                	budget_year int4 NOT NULL,
                                	budget_month int4 NOT NULL,
                                	CONSTRAINT budget_user_id_category_name_budget_year_budget_month_key UNIQUE (user_id, category_name, budget_year, budget_month)
                                )
[2026-02-14 20:02:00] completed in 19 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE savings_goal (
                                	goal_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
                                	user_id int references users(user_id),
                                	goal_name varchar(100) NOT NULL,
                                	target_amount numeric(10, 2) NOT NULL,
                                	current_amount numeric(10, 2) DEFAULT 0.00 NOT NULL,
                                	target_date date,
                                	priority_level int4 DEFAULT 1 NOT NULL,
                                	is_active bool DEFAULT true NOT NULL,
                                	created_at timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL,
                                	CONSTRAINT savings_goal_priority_level_check CHECK (((priority_level >= 1) AND (priority_level <= 5)))
                                )
[2026-02-14 20:02:00] completed in 17 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE financial_transaction (
                                    transaction_id serial PRIMARY KEY, --Same as SQL Server IDENTITY
                                    user_id int REFERENCES users(user_id),
                                    amount numeric(10,2) NOT NULL,
                                    description text,
                                    transaction_date date NOT NULL,
                                    transaction_type varchar(20) CHECK (transaction_type IN ('EXPENSE', 'INCOME'))
                                )
[2026-02-14 20:02:00] completed in 20 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE expense_details (
                                    transaction_id int PRIMARY KEY REFERENCES financial_transaction(transaction_id),
                                    category varchar(100),
                                    payment_method varchar(50),             -- How money left (cash, card, transfer)
                                    category_id int REFERENCES category(category_id),
                                    account_id int REFERENCES account(account_id),
                                    merchant_id int REFERENCES merchant(merchant_id),  -- Who received the money
                                    is_recurring boolean DEFAULT false,        -- Does this repeat monthly?
                                    receipt_url varchar(255)                  -- Proof of purchase
                                )
[2026-02-14 20:02:00] completed in 15 ms
[2026-02-14 20:02:00] uw.money> CREATE TABLE income_details (
                                    transaction_id int PRIMARY KEY REFERENCES financial_transaction(transaction_id),
                                    account_id int REFERENCES account(account_id),
                                    source varchar(100) NOT NULL,              -- Where money came from (employer, client)
                                    income_type varchar(50),                    -- Salary, freelance, investment, gift
                                    is_taxable boolean DEFAULT true,            -- Tax implications
                                    employer_name varchar(100),                 -- For employment income
                                    payment_frequency varchar(20),              -- How often this income occurs
                                    gross_amount numeric(10,2),                 -- Before taxes/deductions
                                    net_amount numeric(10,2)                -- After taxes/deductions
                                )
[2026-02-14 20:02:00] completed in 14 ms


 */