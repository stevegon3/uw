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

-- "money".category definition
CREATE TABLE category (
	category_id serial4 NOT NULL,
	category varchar(100) NULL,
	cat_sub varchar(25) NULL,
	cat_super varchar(25) NULL,
	CONSTRAINT category_pkey PRIMARY KEY (category_id)
);

-- "money".institution definition
CREATE TABLE institution (
	institution_id serial4 NOT NULL,
	institution_name varchar(30) NULL,
	CONSTRAINT institution_pkey PRIMARY KEY (institution_id)
);

-- "money".merchant definition
CREATE TABLE merchant (
	merchant_id serial4 NOT NULL,
	merchant_name varchar(30) NULL,
	merchant_type varchar(30) NULL,
	CONSTRAINT merchant_pkey PRIMARY KEY (merchant_id)
);

-- "money".users definition
CREATE TABLE users (
	user_id serial4 NOT NULL,
	username varchar(50) NOT NULL,
	email varchar(100) NOT NULL,
	first_name varchar(50) NULL,
	last_name varchar(50) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT users_email_key UNIQUE (email),
	CONSTRAINT users_pkey PRIMARY KEY (user_id),
	CONSTRAINT users_username_key UNIQUE (username)
);

-- "money".account definition
CREATE TABLE account (
	account_id serial4 NOT NULL,
	account_name varchar(30) NULL,
	institution_id int4 NULL,
	account_number varchar(30) NULL,
	CONSTRAINT account_pkey PRIMARY KEY (account_id),
	CONSTRAINT account_institution_id_fkey FOREIGN KEY (institution_id) REFERENCES institution(institution_id)
);

-- "money".budget definition
CREATE TABLE budget (
	budget_id serial4 NOT NULL,
	user_id int4 NULL,
	category_name varchar(100) NOT NULL,
	monthly_limit numeric(10, 2) NOT NULL,
	current_spent numeric(10, 2) DEFAULT 0.00 NULL,
	budget_year int4 NOT NULL,
	budget_month int4 NOT NULL,
	CONSTRAINT budget_pkey PRIMARY KEY (budget_id),
	CONSTRAINT budget_user_id_category_name_budget_year_budget_month_key UNIQUE (user_id, category_name, budget_year, budget_month),
	CONSTRAINT budget_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- "money".savings_goal definition
CREATE TABLE savings_goal (
	goal_id serial4 NOT NULL,
	user_id int4 NULL,
	goal_name varchar(100) NOT NULL,
	target_amount numeric(10, 2) NOT NULL,
	current_amount numeric(10, 2) DEFAULT 0.00 NULL,
	target_date date NULL,
	priority_level int4 DEFAULT 1 NULL,
	is_active bool DEFAULT true NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT savings_goal_pkey PRIMARY KEY (goal_id),
	CONSTRAINT savings_goal_priority_level_check CHECK (((priority_level >= 1) AND (priority_level <= 5))),
	CONSTRAINT savings_goal_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Supertype
CREATE TABLE financial_transaction (
    transaction_id serial PRIMARY KEY,
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