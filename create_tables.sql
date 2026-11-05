CREATE DATABASE lanka_fabric_shop;
USE lanka_fabric_shop;

CREATE TABLE sales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sale_date DATE,
    sku VARCHAR(20),
    product_name VARCHAR(100),
    quantity_sold INT,
    unit_price DECIMAL(10,2),
    customer_name VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS inventory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sku VARCHAR(20) UNIQUE,
    product_name VARCHAR(100),
    current_stock INT
);

CREATE TABLE IF NOT EXISTS admins (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);

-- Example admin
INSERT INTO admins (username, password)
VALUES ('admin', SHA2('admin123', 256));

SELECT * FROM admins;