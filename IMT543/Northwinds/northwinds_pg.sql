/*Set DATEFORMAT so that the date strings are SERIALerpreted correctly regardless of
the default DATEFORMAT on the server.*/

DROP TABLE IF EXISTS Order_Details CASCADE;
DROP TABLE IF EXISTS Orders CASCADE;
DROP TABLE IF EXISTS Products CASCADE;
DROP TABLE IF EXISTS Suppliers CASCADE;
DROP TABLE IF EXISTS Shippers CASCADE;
DROP TABLE IF EXISTS Customers CASCADE;
DROP TABLE IF EXISTS Categories CASCADE;
DROP TABLE IF EXISTS Employees CASCADE;

CREATE TABLE Employees (
EmployeeID SERIAL PRIMARY KEY ,
LastName VARCHAR (20) NOT NULL ,
FirstName VARCHAR (10) NOT NULL ,
Title VARCHAR (30) NULL ,
TitleOfCourtesy VARCHAR (25) NULL ,
BirthDate TIMESTAMP NULL ,
HireDate TIMESTAMP NULL ,
Address VARCHAR (60) NULL ,
City VARCHAR (15) NULL ,
Region VARCHAR (15) NULL ,
PostalCode VARCHAR (10) NULL ,
Country VARCHAR (15) NULL ,
HomePhone VARCHAR (24) NULL ,
Extension VARCHAR (4) NULL ,
Photo bytea NULL ,
Notes TEXT NULL ,
ReportsTo int,
PhotoPath VARCHAR (255) NULL,
CONSTRAINT FK_Employees_Employees FOREIGN KEY(ReportsTo) REFERENCES Employees (EmployeeID),
CONSTRAINT CK_Birthdate CHECK (BirthDate < CURRENT_TIMESTAMP)
);
CREATE INDEX idx_LastName ON Employees(LastName);
CREATE INDEX idx_employees_PostalCode ON Employees(PostalCode);

CREATE TABLE Categories (
CategoryID SERIAL PRIMARY KEY ,
CategoryName VARCHAR (15) NOT NULL ,
Description TEXT NULL ,
Picture bytea NULL);
CREATE INDEX idx_CategoryName ON Categories(CategoryName);

CREATE TABLE Customers (
CustomerID CHAR (5) PRIMARY KEY ,
CompanyName VARCHAR (40) NOT NULL ,
ContactName VARCHAR (30) NULL ,
ContactTitle VARCHAR (30) NULL ,
Address VARCHAR (60) NULL ,
City VARCHAR (15) NULL ,
Region VARCHAR (15) NULL ,
PostalCode VARCHAR (10) NULL ,
Country VARCHAR (15) NULL ,
Phone VARCHAR (24) NULL ,
Fax VARCHAR (24) NULL);
CREATE INDEX idx_City ON Customers(City);
CREATE INDEX idx_customers_CompanyName ON Customers(CompanyName);
CREATE INDEX idx_customers_PostalCode ON Customers(PostalCode);
CREATE INDEX idx_Region ON Customers(Region);

CREATE TABLE Shippers (
ShipperID SERIAL PRIMARY KEY ,
CompanyName VARCHAR (40) NOT NULL ,
Phone VARCHAR (24) NULL);

CREATE TABLE Suppliers (
SupplierID SERIAL PRIMARY KEY ,
CompanyName VARCHAR (40) NOT NULL ,
ContactName VARCHAR (30) NULL ,
ContactTitle VARCHAR (30) NULL ,
Address VARCHAR (60) NULL ,
City VARCHAR (15) NULL ,
Region VARCHAR (15) NULL ,
PostalCode VARCHAR (10) NULL ,
Country VARCHAR (15) NULL ,
Phone VARCHAR (24) NULL ,
Fax VARCHAR (24) NULL ,
HomePage TEXT NULL);
CREATE INDEX idx_suppliers_CompanyName ON Suppliers(CompanyName);
CREATE INDEX idx_suppliers_PostalCode ON Suppliers(PostalCode);

CREATE TABLE Orders (
OrderID SERIAL PRIMARY KEY ,
CustomerID CHAR (5) NULL ,
EmployeeID int ,
OrderDate TIMESTAMP NULL ,
RequiredDate TIMESTAMP NULL ,
ShippedDate TIMESTAMP NULL ,
ShipVia int ,
Freight NUMERIC(10,2) NULL DEFAULT 0,
ShipName VARCHAR (40) NULL ,
ShipAddress VARCHAR (60) NULL ,
ShipCity VARCHAR (15) NULL ,
ShipRegion VARCHAR (15) NULL ,
ShipPostalCode VARCHAR (10) NULL ,
ShipCountry VARCHAR (15) NULL,
CONSTRAINT FK_Orders_Customers FOREIGN KEY(CustomerID) REFERENCES Customers (CustomerID),
CONSTRAINT FK_Orders_Employees FOREIGN KEY(EmployeeID) REFERENCES Employees (EmployeeID),
CONSTRAINT FK_Orders_Shippers FOREIGN KEY(ShipVia) REFERENCES Shippers (ShipperID)
                    );
CREATE INDEX idx_CustomerID ON Orders(CustomerID);
CREATE INDEX idx_CustomersOrders ON Orders(CustomerID);
CREATE INDEX idx_EmployeeID ON Orders(EmployeeID);
CREATE INDEX idx_EmployeesOrders ON Orders(EmployeeID);
CREATE INDEX idx_OrderDate ON Orders(OrderDate);
CREATE INDEX idx_ShippedDate ON Orders(ShippedDate);
CREATE INDEX idx_ShippersOrders ON Orders(ShipVia);
CREATE INDEX idx_ShipPostalCode ON Orders(ShipPostalCode);

CREATE TABLE Products (
ProductID SERIAL PRIMARY KEY ,
ProductName VARCHAR (40) NOT NULL ,
SupplierID int ,
CategoryID int ,
QuantityPerUnit VARCHAR (20) NULL ,
UnitPrice NUMERIC(10,2) NULL DEFAULT 0,
UnitsInStock int NULL DEFAULT 0,
UnitsOnOrder int NULL DEFAULT 0,
ReorderLevel int NULL DEFAULT 0,
Discontinued smallint NOT NULL DEFAULT 0,
CONSTRAINT FK_Products_Categories FOREIGN KEY(CategoryID) REFERENCES Categories (CategoryID),
CONSTRAINT FK_Products_Suppliers FOREIGN KEY(SupplierID) REFERENCES Suppliers (SupplierID),
CONSTRAINT CK_Products_UnitPrice CHECK (UnitPrice >= 0),
CONSTRAINT CK_ReorderLevel CHECK (ReorderLevel >= 0),
CONSTRAINT CK_UnitsInStock CHECK (UnitsInStock >= 0),
CONSTRAINT CK_UnitsOnOrder CHECK (UnitsOnOrder >= 0)
);
CREATE INDEX idx_CategoriesProducts ON Products(CategoryID);
CREATE INDEX idx_CategoryID ON Products(CategoryID);
CREATE INDEX idx_ProductName ON Products(ProductName);
CREATE INDEX idx_SupplierID ON Products(SupplierID);
CREATE INDEX idx_SuppliersProducts ON Products(SupplierID);

CREATE TABLE Order_Details (
OrderID SERIAL NOT NULL ,
ProductID int NOT NULL ,
UnitPrice NUMERIC(10,2) NOT NULL DEFAULT 0,
Quantity smallint NOT NULL DEFAULT 1,
Discount real NOT NULL DEFAULT 0,
CONSTRAINT PK_Order_Details PRIMARY KEY(OrderID,ProductID),
CONSTRAINT FK_Order_Details_Orders FOREIGN KEY(OrderID) REFERENCES Orders (OrderID),
CONSTRAINT FK_Order_Details_Products FOREIGN KEY(ProductID) REFERENCES Products (ProductID),
CONSTRAINT CK_Discount CHECK (Discount >= 0 and (Discount <= 1)),
CONSTRAINT CK_Quantity CHECK (Quantity > 0),
CONSTRAINT CK_UnitPrice CHECK (UnitPrice >= 0)
);
CREATE INDEX idx_OrderID ON Order_Details(OrderID);
CREATE INDEX idx_OrdersOrder_Details ON Order_Details(OrderID);
CREATE INDEX idx_ProductID ON Order_Details(ProductID);
CREATE INDEX idx_ProductsOrder_Details ON Order_Details(ProductID);

