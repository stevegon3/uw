CREATE OR REPLACE VIEW Customer_and_Suppliers_by_City AS
SELECT City, CompanyName, ContactName, 'Customers' AS Relationship
FROM Customers
UNION SELECT City, CompanyName, ContactName, 'Suppliers'
FROM Suppliers;

CREATE OR REPLACE VIEW Alphabetical_list_of_products AS
SELECT Products.*, Categories.CategoryName
FROM Categories INNER JOIN Products ON Categories.CategoryID = Products.CategoryID
WHERE (((Products.Discontinued)=0));

CREATE OR REPLACE VIEW Current_Product_List AS
SELECT Product_List.ProductID, Product_List.ProductName
FROM Products AS Product_List
WHERE (((Product_List.Discontinued)=0));

CREATE OR REPLACE VIEW Orders_Qry AS
SELECT Orders.OrderID, Orders.CustomerID, Orders.EmployeeID, Orders.OrderDate, Orders.RequiredDate,
Orders.ShippedDate, Orders.ShipVia, Orders.Freight, Orders.ShipName, Orders.ShipAddress, Orders.ShipCity,
Orders.ShipRegion, Orders.ShipPostalCode, Orders.ShipCountry,
Customers.CompanyName, Customers.Address, Customers.City, Customers.Region, Customers.PostalCode, Customers.Country
FROM Customers INNER JOIN Orders ON Customers.CustomerID = Orders.CustomerID;

CREATE OR REPLACE VIEW Products_Above_Average_Price AS
SELECT Products.ProductName, Products.UnitPrice
FROM Products
WHERE Products.UnitPrice>(SELECT AVG(UnitPrice) From Products);

CREATE OR REPLACE VIEW Products_by_Category AS
SELECT Categories.CategoryName, Products.ProductName, Products.QuantityPerUnit, Products.UnitsInStock, Products.Discontinued
FROM Categories INNER JOIN Products ON Categories.CategoryID = Products.CategoryID
WHERE Products.Discontinued <> 1;

CREATE OR REPLACE VIEW Quarterly_Orders AS
SELECT DISTINCT Customers.CustomerID, Customers.CompanyName, Customers.City, Customers.Country
FROM Customers RIGHT JOIN Orders ON Customers.CustomerID = Orders.CustomerID
WHERE Orders.OrderDate BETWEEN '19970101' And '19971231';

CREATE OR REPLACE VIEW Invoices AS
SELECT Orders.ShipName, Orders.ShipAddress, Orders.ShipCity, Orders.ShipRegion, Orders.ShipPostalCode,
Orders.ShipCountry, Orders.CustomerID, Customers.CompanyName AS CustomerName, Customers.Address, Customers.City,
Customers.Region, Customers.PostalCode, Customers.Country,
(FirstName || ' ' || LastName) AS Salesperson,
Orders.OrderID, Orders.OrderDate, Orders.RequiredDate, Orders.ShippedDate, Shippers.CompanyName As ShipperName,
Order_Details.ProductID, Products.ProductName, Order_Details.UnitPrice, Order_Details.Quantity,
Order_Details.Discount,
(CAST((Order_Details.UnitPrice*Quantity*(1-Discount)/100)*100 AS NUMERIC(10,2))) AS ExtendedPrice, Orders.Freight
FROM 	Shippers INNER JOIN
(Products INNER JOIN
(
(Employees INNER JOIN
(Customers INNER JOIN Orders ON Customers.CustomerID = Orders.CustomerID)
ON Employees.EmployeeID = Orders.EmployeeID)
INNER JOIN Order_Details ON Orders.OrderID = Order_Details.OrderID)
ON Products.ProductID = Order_Details.ProductID)
ON Shippers.ShipperID = Orders.ShipVia;

CREATE OR REPLACE VIEW Order_Details_Extended AS
SELECT Order_Details.OrderID, Order_Details.ProductID, Products.ProductName,
Order_Details.UnitPrice, Order_Details.Quantity, Order_Details.Discount,
(CAST((Order_Details.UnitPrice*Quantity*(1-Discount)/100)*100 AS NUMERIC(10,2))) AS ExtendedPrice
FROM Products INNER JOIN Order_Details ON Products.ProductID = Order_Details.ProductID;

CREATE OR REPLACE VIEW Order_Subtotals AS
SELECT Order_Details.OrderID, Sum(CAST((Order_Details.UnitPrice*Quantity*(1-Discount)/100)*100 AS NUMERIC(10,2))) AS Subtotal
FROM Order_Details
GROUP BY Order_Details.OrderID;

CREATE OR REPLACE VIEW Product_Sales_for_1997 AS
SELECT Categories.CategoryName, Products.ProductName,
Sum(CAST((Order_Details.UnitPrice*Quantity*(1-Discount)/100)*100 AS NUMERIC(10,2))) AS ProductSales
FROM (Categories INNER JOIN Products ON Categories.CategoryID = Products.CategoryID)
INNER JOIN (Orders
INNER JOIN Order_Details ON Orders.OrderID = Order_Details.OrderID)
ON Products.ProductID = Order_Details.ProductID
WHERE (((Orders.ShippedDate) Between '19970101' And '19971231'))
GROUP BY Categories.CategoryName, Products.ProductName;

CREATE OR REPLACE VIEW Category_Sales_for_1997 AS
SELECT Product_Sales_for_1997.CategoryName, Sum(Product_Sales_for_1997.ProductSales) AS CategorySales
FROM Product_Sales_for_1997
GROUP BY Product_Sales_for_1997.CategoryName;

CREATE OR REPLACE VIEW Sales_by_Category AS
SELECT Categories.CategoryID, Categories.CategoryName, Products.ProductName,
Sum(Order_Details_Extended.ExtendedPrice) AS ProductSales
FROM 	Categories INNER JOIN
(Products INNER JOIN
(Orders INNER JOIN Order_Details_Extended ON Orders.OrderID = Order_Details_Extended.OrderID)
ON Products.ProductID = Order_Details_Extended.ProductID)
ON Categories.CategoryID = Products.CategoryID
WHERE Orders.OrderDate BETWEEN '19970101' And '19971231'
GROUP BY Categories.CategoryID, Categories.CategoryName, Products.ProductName;

CREATE OR REPLACE VIEW Sales_Totals_by_Amount AS
SELECT Order_Subtotals.Subtotal AS SaleAmount, Orders.OrderID, Customers.CompanyName, Orders.ShippedDate
FROM 	Customers INNER JOIN
(Orders INNER JOIN Order_Subtotals ON Orders.OrderID = Order_Subtotals.OrderID)
ON Customers.CustomerID = Orders.CustomerID
WHERE (Order_Subtotals.Subtotal >2500) AND (Orders.ShippedDate BETWEEN '19970101' And '19971231');

CREATE OR REPLACE VIEW Summary_of_Sales_by_Quarter AS
SELECT Orders.ShippedDate, Orders.OrderID, Order_Subtotals.Subtotal
FROM Orders INNER JOIN Order_Subtotals ON Orders.OrderID = Order_Subtotals.OrderID
WHERE Orders.ShippedDate IS NOT NULL;

CREATE OR REPLACE VIEW Summary_of_Sales_by_Year AS
SELECT Orders.ShippedDate, Orders.OrderID, Order_Subtotals.Subtotal
FROM Orders INNER JOIN Order_Subtotals ON Orders.OrderID = Order_Subtotals.OrderID
WHERE Orders.ShippedDate IS NOT NULL;
