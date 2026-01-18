/*-- Create custom types
CREATE DOMAIN id AS VARCHAR(11) NOT NULL;
CREATE DOMAIN tid AS VARCHAR(6) NOT NULL;
CREATE DOMAIN empid AS CHAR(9) NOT NULL;
*/
-- Drop all tables if they exist
DROP TABLE IF EXISTS employee CASCADE;
DROP TABLE IF EXISTS pub_info CASCADE;
DROP TABLE IF EXISTS discounts CASCADE;
DROP TABLE IF EXISTS roysched CASCADE;
DROP TABLE IF EXISTS sales CASCADE;
DROP TABLE IF EXISTS titleauthor CASCADE;
DROP TABLE IF EXISTS titles CASCADE;
DROP TABLE IF EXISTS jobs CASCADE;
DROP TABLE IF EXISTS stores CASCADE;
DROP TABLE IF EXISTS publishers CASCADE;
DROP TABLE IF EXISTS authors CASCADE;

-- Create tables
CREATE TABLE authors (
    au_id varchar(12) primary key,
    au_lname VARCHAR(40) NOT NULL,
    au_fname VARCHAR(20) NOT NULL,
    phone CHAR(12) NOT NULL DEFAULT 'UNKNOWN',
    address VARCHAR(40),
    city VARCHAR(20),
    state CHAR(2),
    zip CHAR(5)
        CHECK (zip ~ '^[0-9]{5}$'),
    contract BOOLEAN NOT NULL
);

CREATE TABLE publishers (
    pub_id CHAR(4) NOT NULL PRIMARY KEY,
    pub_name VARCHAR(40),
    city VARCHAR(20),
    state CHAR(2),
    country VARCHAR(30) DEFAULT 'USA'
);

CREATE TABLE titles (
    title_id VARCHAR(6) NOT NULL PRIMARY KEY,
    title VARCHAR(80) NOT NULL,
    type CHAR(12) NOT NULL DEFAULT 'UNDECIDED',
    pub_id CHAR(4) REFERENCES publishers(pub_id),
    price NUMERIC(10,2),
    advance NUMERIC(10,2),
    royalty INTEGER,
    ytd_sales INTEGER,
    notes VARCHAR(200),
    pubdate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE titleauthor (
    au_id varchar(12) REFERENCES authors(au_id),
    title_id varchar(6)  REFERENCES titles(title_id),
    au_ord SMALLINT,
    royaltyper INTEGER,
    CONSTRAINT pk_titleauthor PRIMARY KEY (au_id, title_id)
);

CREATE TABLE stores (
    stor_id CHAR(4) NOT NULL PRIMARY KEY,
    stor_name VARCHAR(40),
    stor_address VARCHAR(40),
    city VARCHAR(20),
    state CHAR(2),
    zip CHAR(5)
);

CREATE TABLE sales (
    stor_id CHAR(4) NOT NULL REFERENCES stores(stor_id),
    ord_num VARCHAR(20) NOT NULL,
    ord_date TIMESTAMP NOT NULL,
    qty SMALLINT NOT NULL,
    payterms VARCHAR(12) NOT NULL,
    title_id varchar(6)  REFERENCES titles(title_id),
    CONSTRAINT pk_sales PRIMARY KEY (stor_id, ord_num, title_id)
);

CREATE TABLE roysched (
    title_id VARCHAR(6) NOT NULL REFERENCES titles(title_id),
    lorange INTEGER,
    hirange INTEGER,
    royalty INTEGER
);

CREATE TABLE discounts (
    discounttype VARCHAR(40) PRIMARY KEY,
    stor_id CHAR(4) REFERENCES stores(stor_id),
    lowqty SMALLINT,
    highqty SMALLINT,
    discount NUMERIC(4,2) NOT NULL
);

CREATE TABLE jobs (
    job_id SERIAL PRIMARY KEY,
    job_desc VARCHAR(50) NOT NULL DEFAULT 'New Position - title not formalized yet',
    min_lvl SMALLINT NOT NULL CHECK (min_lvl >= 10),
    max_lvl SMALLINT NOT NULL CHECK (max_lvl <= 250)
);

CREATE TABLE pub_info (
    pub_id CHAR(4) NOT NULL
        REFERENCES publishers(pub_id)
        CONSTRAINT pk_pub_info PRIMARY KEY,
    logo BYTEA,
    pr_info TEXT
);

CREATE TABLE employee (
    emp_id CHAR(9) NOT NULL PRIMARY KEY,
    fname VARCHAR(20) NOT NULL,
    minit CHAR(1),
    lname VARCHAR(30) NOT NULL,
    job_id SMALLINT NOT NULL DEFAULT 1
        REFERENCES jobs(job_id),
    job_lvl SMALLINT DEFAULT 10,
    pub_id CHAR(4) NOT NULL DEFAULT '9952'
        REFERENCES publishers(pub_id),
    hire_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_employee_name ON employee(lname, fname, minit);
CREATE INDEX idx_authors_name ON authors(au_lname, au_fname);
CREATE INDEX idx_sales_title_id ON sales(title_id);
CREATE INDEX idx_titles_title ON titles(title);
CREATE INDEX idx_titleauthor_au_id ON titleauthor(au_id);
CREATE INDEX idx_titleauthor_title_id ON titleauthor(title_id);
CREATE INDEX idx_roysched_title_id ON roysched(title_id);

-- Create view
CREATE OR REPLACE VIEW titleview AS
SELECT title, au_ord, au_lname, price, ytd_sales, pub_id
FROM authors, titles, titleauthor
WHERE authors.au_id = titleauthor.au_id
   AND titles.title_id = titleauthor.title_id;

/*
-- Create functions
CREATE OR REPLACE FUNCTION employee_insupd_trigger()
RETURNS TRIGGER AS $$
DECLARE
    min_lvl SMALLINT;
    max_lvl SMALLINT;
    emp_lvl SMALLINT;
    job_id SMALLINT;
BEGIN
    -- Get the range of level for this job type from the jobs table
    SELECT j.min_lvl, j.max_lvl, NEW.job_lvl, NEW.job_id
    INTO min_lvl, max_lvl, emp_lvl, job_id
    FROM jobs j
    WHERE j.job_id = NEW.job_id;

    IF (job_id = 1) AND (emp_lvl <> 10) THEN
        RAISE EXCEPTION 'Job id 1 expects the default level of 10.';
    ELSIF (emp_lvl < min_lvl OR emp_lvl > max_lvl) THEN
        RAISE EXCEPTION 'The level for job_id:% should be between % and %.', 
            job_id, min_lvl, max_lvl;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER employee_insupd
BEFORE INSERT OR UPDATE ON employee
FOR EACH ROW EXECUTE FUNCTION employee_insupd_trigger();

-- Create stored procedures
CREATE OR REPLACE FUNCTION byroyalty(percentage INTEGER)
RETURNS TABLE(au_id VARCHAR(11)) AS $$
BEGIN
    RETURN QUERY SELECT titleauthor.au_id 
    FROM titleauthor
    WHERE titleauthor.royaltyper = percentage;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION reptq1()
RETURNS TABLE(pub_id TEXT, avg_price NUMERIC) AS $$
BEGIN
    RETURN QUERY 
    SELECT 
        COALESCE(pub_id::TEXT, 'ALL') as pub_id,
        AVG(price)::NUMERIC(10,2) as avg_price
    FROM titles
    WHERE price IS NOT NULL
    GROUP BY ROLLUP(pub_id)
    ORDER BY pub_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION reptq2()
RETURNS TABLE(type TEXT, pub_id TEXT, avg_ytd_sales NUMERIC) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(type, 'ALL') as type,
        COALESCE(pub_id::TEXT, 'ALL') as pub_id,
        AVG(ytd_sales)::NUMERIC(10,2) as avg_ytd_sales
    FROM titles
    WHERE pub_id IS NOT NULL
    GROUP BY ROLLUP(pub_id, type)
    ORDER BY type, pub_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION reptq3(lolimit NUMERIC, hilimit NUMERIC, p_type VARCHAR(12))
RETURNS TABLE(pub_id TEXT, type TEXT, cnt BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(pub_id::TEXT, 'ALL') as pub_id,
        COALESCE(type, 'ALL') as type,
        COUNT(title_id)::BIGINT as cnt
    FROM titles
    WHERE (price > lolimit AND price < hilimit AND type = p_type) 
       OR type LIKE '%cook%'
    GROUP BY ROLLUP(pub_id, type)
    ORDER BY pub_id, type;
END;
$$ LANGUAGE plpgsql;

-- Note: The data insertion part is not included in this conversion as it's quite large
-- and follows similar patterns. The main differences would be:
-- 1. Using single quotes for string literals
-- 2. Adjusting date formats to use ISO format (YYYY-MM-DD)
-- 3. Removing square brackets from identifiers
-- 4. Adjusting money literals to numeric
-- 5. Using FALSE/TRUE for bit fields
-- 6. Converting image data to BYTEA format
-- 7. Using E'...' for escape sequences in strings


     */
