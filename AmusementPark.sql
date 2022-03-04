-- CREATING DATABASE - 

CREATE DATABASE IF NOT EXISTS DBMS_CASE_STUDY ;
USE DBMS_CASE_STUDY ;

-- CREATING TABLES - 

create table IF NOT EXISTS Food_Concessions 
(
foodStand_ID INTEGER PRIMARY KEY, 
worker_ID integer, 
profit decimal, 
food_Type text, 
quantity integer, 
food_bought_time DATETIME 
) ;

create table IF NOT EXISTS Customers 
(
customer_ID integer PRIMARY KEY, 
entry_time DATETIME,
exit_time DATETIME , 
ticketsBought integer, 
fname_customer text, 
lname_customer text 
) ;

create table IF NOT EXISTS Sections_Of_Park 
(
section_ID integer PRIMARY KEY, 
ride_ID int, 
foodStand_ID int, 
theme text, 
area text, 
capacity integer 
);

create table IF NOT EXISTS Rides 
(
ride_ID integer PRIMARY KEY, 
length_of_line int, 
worker_ID integer, 
type_ride text, 
customer_ID integer, 
riding_time datetime 
);

create table IF NOT EXISTS Worker 
(
worker_ID integer PRIMARY KEY, 
fname_worker text, 
lname_worker text, 
wage decimal, 
hours_worked decimal 
) ;

-- ENTERING VALUES - 

INSERT INTO Food_Concessions values(152,100,10.5,'salad',4,'2019-01-31 09:30:25');
INSERT INTO Food_Concessions values(160,102,11.5,'pizza',4,'2019-01-31 19:04:34');
INSERT INTO Food_Concessions values(156,103,13.5,'icecream',4,'2019-01-31 10:09:07');
INSERT INTO Food_Concessions values(168,104,17.5,'taco',4,'2019-01-31 16:18:08');
INSERT INTO Food_Concessions values(189,125,12.7,'Chicken wings',4,'2019-01-31 18:09:57');

INSERT INTO Customers values(2001,'2019-01-31 11:09:15','2019-01-31 14:09:15',5,'Namann','Bhan');
INSERT INTO Customers values(2050,'2019-01-31 12:04:50','2019-01-31 16:09:15',2,'Vedik','Gupta');
INSERT INTO Customers values(2078,'2019-01-31 10:30:21','2019-01-31 15:09:15',4,'Pratyush','Patro');
INSERT INTO Customers values(2008,'2019-01-31 16:31:56','2019-01-31 20:09:15',1,'Vinay','Hariya');
INSERT INTO Customers values(2058,'2019-01-31 17:45:52','2019-01-31 19:09:15',7,'Anmol','Bhatia');

INSERT INTO Customers(customer_ID,entry_time,ticketsBought,fname_customer,lname_customer)
values(2059,'2019-01-31 18:45:52',1,'Rohan','Mehra');
INSERT INTO Customers(customer_ID,entry_time,ticketsBought,fname_customer,lname_customer)
values(2055,'2019-01-31 18:46:02',3,'Karan','Mehta');
INSERT INTO Customers(customer_ID,entry_time,ticketsBought,fname_customer,lname_customer)
values(2077,'2019-01-31 18:48:12',2,'Shabeer','Sheikh');
INSERT INTO Customers(customer_ID,entry_time,ticketsBought,fname_customer,lname_customer)
values(2099,'2019-01-31 18:55:12',2,'Samay','Rao');
INSERT INTO Customers(customer_ID,entry_time,ticketsBought,fname_customer,lname_customer)
values(2098,'2019-01-31 18:56:17',2,'Bhuvan','Bhan');
INSERT INTO Customers(customer_ID,entry_time,ticketsBought,fname_customer,lname_customer)
values(2095,'2019-01-31 18:57:17',2,'Calvin','Pinto');

INSERT INTO Sections_Of_Park values(52,34,152,'Moroccon nights','adult','50');
INSERT INTO Sections_Of_Park values(78,37,160,'Christmas','roller coster','80');
INSERT INTO Sections_Of_Park values(62,26,156,'Snow world','children','30');
INSERT INTO Sections_Of_Park values(89,20,168,'Water Park','adult','10');
INSERT INTO Sections_Of_Park values(97,17,189,'Water Park','children','40');

INSERT INTO Rides values(34,40,100,'Gravitron',2001,'2019-01-31 10:33:01');
INSERT INTO Rides values(37,50,102,'Spinning Teacup',2050,'2019-01-31 16:21:39');
INSERT INTO Rides values(26,20,103,'Thunder',2078,'2019-01-31 17:06:12');
INSERT INTO Rides values(20,15,104,'Nitro',2008,'2019-01-31 18:19:31');
INSERT INTO Rides values(17,35,125,'Laser Tag',2058,'2019-01-31 15:50:47');

insert into Worker values (100, 'Virat', 'kohli', 10.25, 35 );
insert into Worker values (102, 'Phillipe', 'Smith', 8.25, 30 );
insert into Worker values (103, 'Harman', 'Singh', 9.25, 15 );
insert into Worker values (104, 'Harpreet', 'Chabba', 11.00, 35 );
insert into Worker values (125, 'Abish', 'Mathew', 11.25, 29 );

-- PRINTING TABLES - 

select * from Food_Concessions ;
select * from Customers ;
select * from Sections_Of_Park ;
select * from Rides ;
select * from Worker ;

/* 
CREATING A TRIGGER WHICH GETS INVOKED WHEN A NEW CUSTOMER WANTS TO ENTER INTO THE AMUSEMENT PARK .
SO IT WILL CHECK IF THE CAPACITY OF THE PARK IS FULL , OR THERE IS SPACE FOR THE CUSTOMER TO GO . 
IN EITHER CASE IT DISPLAYS THE APPROPRIATE MESSAGE . 
*/

DELIMITER //
CREATE TRIGGER capacity_check 
BEFORE INSERT
ON Customers
FOR EACH ROW
BEGIN
DECLARE capacity INT DEFAULT 0 ;
SET @result = "" ;
SET @total = (select count(*) FROM Customers) ;
SET @already_exited = (select count(exit_time) FROM Customers) ;
SET capacity = 5 ;
IF capacity > @total - @already_exited THEN 
SET @result = "you can go in ! " ;
ELSEIF capacity <= @total - @already_exited THEN
SET @result = "park is full , wait for sometime ! " ;
END IF ;
END //
DELIMITER ;

SELECT @result ; 

/* 
CREATING A TABLE WHICH WILL STORE THE DIFFERENCE OF THE NUMBER OF HOURS WORKED BY A WORKER IN A WEEK 
AND THE MINIMUM STANDARD NUMBER OF HOURS TO BE WORKED IN A WEEK , WHICH IS CONSIDERED AS 33 . 
THIS CALCULATION IS DONE IN AN AFTER INSERT TRIGGER BELOW WHICH GETS INVOKED AFTER ANY ROW IS INSERTED
IN THE Worker TABLE
*/

CREATE TABLE IF NOT EXISTS workerhoursdifference(
worker_ID int ,
fname text, 
lname text, 
hours_workedDIFF decimal 
);

DELIMITER $$
CREATE TRIGGER HOURSWORKEDDIFFERENCE
    AFTER INSERT
    ON Worker FOR EACH ROW
BEGIN
    DECLARE difference decimal;
    SET difference = NEW.hours_worked-33;
    INSERT INTO workerhoursdifference VALUES(NEW.worker_ID ,NEW.fname_worker,NEW.lname_worker,difference);
END$$    

DELIMITER ;