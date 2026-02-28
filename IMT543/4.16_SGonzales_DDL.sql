-- Drop tables in reverse dependency order with CASCADE
DROP TABLE IF EXISTS "transaction" CASCADE;
DROP TABLE IF EXISTS expense CASCADE;
DROP TABLE IF EXISTS income CASCADE;
DROP TABLE IF EXISTS budget CASCADE;
DROP TABLE IF EXISTS savings_goal CASCADE;
DROP TABLE IF EXISTS account CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS merchant CASCADE;
DROP TABLE IF EXISTS "rule" CASCADE;
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

-- "money"."rule" definition
CREATE TABLE "rule" (
	rule_id serial4 NOT NULL,
	rule_name varchar(50) NULL,
	payee varchar(150) NULL,
	payee_new varchar(50) NULL,
	cat_new varchar(100) NULL,
	CONSTRAINT rule_pkey PRIMARY KEY (rule_id)
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

-- "money".expense definition
CREATE TABLE expense (
	expense_id serial4 NOT NULL,
	user_id int4 NULL,
	amount numeric(10, 2) NOT NULL,
	description text NULL,
	expense_date date NOT NULL,
	category varchar(100) NOT NULL,
	payment_method varchar(50) NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT expense_pkey PRIMARY KEY (expense_id),
	CONSTRAINT expense_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- "money".income definition
CREATE TABLE income (
	income_id serial4 NOT NULL,
	user_id int4 NULL,
	amount numeric(10, 2) NOT NULL,
	"source" varchar(100) NOT NULL,
	income_date date NOT NULL,
	income_type varchar(50) DEFAULT 'salary'::character varying NULL,
	description text NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT income_pkey PRIMARY KEY (income_id),
	CONSTRAINT income_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
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

-- "money"."transaction" definition
CREATE TABLE "transaction" (
	transaction_id serial4 NOT NULL,
	merchant varchar(100) NULL,
	transaction_dt date NULL,
	category_id int4 NULL,
	amt float8 NULL,
	account_id int4 NULL,
	merchant_id int4 NULL,
	rule_id int4 NULL,
	CONSTRAINT transaction_account_id_fkey FOREIGN KEY (account_id) REFERENCES account(account_id),
	CONSTRAINT transaction_category_id_fkey FOREIGN KEY (category_id) REFERENCES category(category_id),
	CONSTRAINT transaction_merchant_id_fkey FOREIGN KEY (merchant_id) REFERENCES merchant(merchant_id),
	CONSTRAINT transaction_rule_id_fkey FOREIGN KEY (rule_id) REFERENCES "rule"(rule_id)
);