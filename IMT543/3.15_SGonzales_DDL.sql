CREATE SCHEMA IF NOT EXISTS money;

DROP TABLE IF EXISTS money.institution CASCADE;
CREATE TABLE institution (
  institution_id  serial PRIMARY KEY,
  institution_name  VARCHAR(30));

DROP TABLE IF EXISTS money.account CASCADE;
CREATE TABLE account (
  account_id  serial PRIMARY KEY,
  account_name  VARCHAR(30),
  institution_id int REFERENCES institution(institution_id),
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
CREATE TABLE merchant (
  merchant_id  serial PRIMARY KEY,
  merchant_name  VARCHAR(30),
  merchant_type varchar(30));

DROP TABLE IF EXISTS money.transaction CASCADE;
CREATE TABLE IF NOT EXISTS money.transaction (
    transaction_id      SERIAL PRIMARY KEY,
    merchant            VARCHAR(100),
    transaction_dt      DATE,
    category_id         int REFERENCES category(category_id),
    amt                 FLOAT,
    account_id          int REFERENCES account(account_id),
    merchant_id         int REFERENCES merchant(merchant_id),
    rule_id             int REFERENCES rule(rule_id));

SELECT * FROM account;
SELECT * FROM institution;

SELECT * FROM authors;
ALTER TABLE authors DROP COLUMN au_mname;
CREATE TABLE authors AS SELECT au_id, au_lname, au_fname, phone, address, city, state, zip, contract FROM authors_bak;
DROP TABLE authors_bak;