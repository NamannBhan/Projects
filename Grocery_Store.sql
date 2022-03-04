import mysql.connector
import sys
from tabulate import tabulate
def registration():
    mydb1=mysql.connector.connect(host="localhost",database="test",user="root",passwd="m@ndate22")
    mycursor1=mydb1.cursor()
    name=input("Enter your username --> ")
    mycursor1.execute("select * from users where username = %s",(name,))
    p=mycursor1.fetchall()
    if p != []:
        print("Username already exists.")
        registration()
    pass1=input("Enter a password --> ")
    pass2=input("Retype Password --> ")
    if pass1 != pass2:
        print("Passwords do not match")
        registration()
    else:
        ques1=input("Security Question 1: Name of your First Pet -->")
        ques2=input("Security Question 2: Name of your First Mobile Phone brand -->")
        sql = "INSERT INTO users VALUES (%s, %s, %s, %s)"
        val = (name, pass1, ques1, ques2)
        mycursor1.execute(sql,val)
        mydb1.commit()
        print(mycursor1.rowcount,"Account creation successful")
        login()
def login():
    mydb=mysql.connector.connect(host="localhost",database="test",user="root",passwd="m@ndate22")
    mycursor=mydb.cursor()
    choice=input("1- Register       \n2- Login       \n3- Reset Password \n4-Exit\n")
    if choice=="1":
        registration()
    elif choice=="2":
        user_name=input("enter your username --> ")
        password=input("enter your password --> ")

        mycursor.execute("select * from users")
        p=mycursor.fetchall()
        flag=0
        for i in p :
            if user_name == i[0]:
                flag=1
                if password == i[1]:
                    print("login successful")
                    print("Welcome ",user_name)
                    del_status(user_name)
                    break;
                else:
                    print("Incorrect Password. Re-enter Password")
                    login()
        if flag==0:
            print("Username not found")
            login()
    elif choice=='3':
        reset()
    elif choice=='4':
        sys.exit()
    else:
        print("Choice not identified. Please try again.")
        login()

def reset():
    mydb2=mysql.connector.connect(host="localhost",database="test",user="root",passwd="m@ndate22")
    mycursor2=mydb2.cursor()
    name=input("Enter your username --> ")
    mycursor2.execute("select * from users")
    p=mycursor2.fetchall()
    flag=0
    for i in p :
        if name == i[0]:
            flag=1
            ans1=input("Name of your First Pet --> ")
            if ans1.upper() == i[2].upper():
                ans2=input("Name of your First Mobile Phone brand --> ")
                if ans2.upper() == i[3].upper():
                    resetpwd(name)
                    break3
                else:
                    print("Answer is incorrect")
                    reset()
            else:
                print("Answer is incorrect")
                reset()
    if flag==0:
        print("Username not found")
        reset()

def resetpwd(name):
    mydb3=mysql.connector.connect(host="localhost",database="test",user="root",passwd="m@ndate22")
    mycursor3=mydb3.cursor()
    pwd1=input('Enter Password -->')
    pwd2=input("Re-enter Password -->")
    if pwd1 != pwd2:
        print("Passwords do not match")
        resetpwd(name)
    else:
        sql = "UPDATE users SET pwd = (%s) WHERE username = (%s)"
        val = (pwd1,name)
        mycursor3.execute(sql,val)

        mydb3.commit()

        print(mycursor3.rowcount, "record(s) affected")

    login()

def del_status(name):
    mydb4=mysql.connector.connect(host="localhost",database="test",user="root",passwd="m@ndate22")
    mycursor4=mydb4.cursor()
    ch=input("\n1-Place Order\n2-View Orders \n3-Cancel Order\n4-Logout\n")
    if ch=='1':
        place_order(name)
    elif ch=='2':
        mycursor4.execute("select * from grocery where username = %s",(name,))
        g=mycursor4.fetchall()
        data=list(g)
        if g == []:
            print("\n \nNo items orderd.")
        else:
            print (tabulate(data, headers=["Name","Item", "Delivery Status"]))
        del_status(name)
    elif ch=='3':
        can_order(name)
    elif ch=='4':
        print("Thank you")
        login()
    else:
        print("Choice not recognized.")
        del_status(name)

def place_order(name):
    mydb5=mysql.connector.connect(host="localhost",database="test",user="root",passwd="m@ndate22")
    mycursor5=mydb5.cursor()
    print("From the Grocery handout enter the code of the grocery item you want.\nTo Exit Enter 1.")
    code='0'
    while(code!='1'):
        code=input("Enter Code --> ")
        if code=='1':
            continue
        mycursor5.execute("select * from grocery_items where code = %s",(code,))
        p=mycursor5.fetchone()
        if p == []:
            print("Code Doesn't exists.")
            continue
        print (p[1])
        sql = "INSERT INTO grocery VALUES (%s,%s,'Not Delivered')"
        val = (name, p[1],)
        mycursor5.execute(sql,val)
        mydb5.commit()
    del_status(name)

def can_order(name):
    mydb6=mysql.connector.connect(host="localhost",database="test",user="root",passwd="m@ndate22")
    mycursor6=mydb6.cursor()
    mycursor6.execute("select * from grocery where username = %s and delivery_status= 'Not Delivered'",(name,))
    k=mycursor6.fetchall()
    data=list(k)
    if k == []:
        print("\n \nNo items remaining.")
        flag=1
    else:
        print("The Orders that can be cancelled are:\n")
        print (tabulate(data, headers=["Name","Item", "Delivery Status"]))
        item='0'
        while(item!='1'):
            item=input("Enter Name of item/nEnter 1 to exit -->")
            if item=='1':
                continue
            else:
                sql = "DELETE FROM grocery WHERE items = %s"
                mycursor6.execute(sql,(item,))
                mydb6.commit()
                print(mycursor6.rowcount, "record(s) deleted")
    del_status(name)



login()

'0001','rice'
'0002','wheat flour'
'0003','gram seeds'
'0004','salt'
'0005','sugar'
'0006','chashew'
'0007','almond'
'0008','pepper'
'0009','kethup'
'0010','bornvitta'
'0011','horlicks'
'0012','tea'
'0013','cofee'
'0014','magie'
'0015','chips'
'0016','soup'
'0017','detergent'
'0018','harpic'
'0019','loofah'
'0020','facewash'
'0021','facepack'
'0022','kala hit'
'0023','goodnight mosquito killer'
'0024','dal'
'0025','daadar'
'0026','toothbursh'
'0027','tooth brush'
'0028','tooth paste'
'0029','vicks'
'0030','butter'
'0031','Dettol'
'0032','curd'
'0033','butter'
'0034','cheese'
'0035','ghee'
'0036','oil'
'0037','rohafza'
'0038','salted peanuts'
'0039','dry fruits'
'0040','paneer'
'0041','buscuits'
'0042','chocolate'
'0043','chocolate syrup'
'0044','baking soda'
'0045','vanilla essence'
'0046','milk'
'0047','cookies'
'0048','toast'
'0049','papad'
'0050','hing'

Create database test;

use test;
create table users(username varchar(20),pwd varchar(16),ques1 varchar(50),ques2 varchar(20));

insert into users values("Mayank","1223","None", "Lenovo");
insert into users values("Naman","2212","Bruno","Oneplus");

select * from users;

select pwd from users where username="Mayank";

create table grocery(
username varchar(50),
items varchar(50),
delivery_status varchar(50)
);
insert into grocery VALUES("Naman","Rice","Delivered");
insert into grocery VALUES("Vedik","Banana","Not delivered");
insert into grocery VALUES("Mayank","Waffles","Delivered");
insert into grocery VALUES("Mayank","Ice-cream","Not delivered");
insert into grocery VALUES("Vedik","Cadbury Silk","Not delivered");
insert Into grocery values("Naman","Onions","Delivered");
select * from grocery;
drop table grocery;
create table grocery_items(code varchar(50) primary key, item varchar(50));
insert into grocery_items values("0001","rice");
insert into grocery_items values("0002","wheat flour");
insert into grocery_items values("0003","gram seeds");
insert into grocery_items values("0004","salt");
insert into grocery_items values("0005","sugar");

insert into grocery_items values("0006","chashew");

insert into grocery_items values("0007","almond");

insert into grocery_items values("0008","pepper");

insert into grocery_items values("0009","kethup");

insert into grocery_items values("0010","bornvitta");

insert into grocery_items values("0011","horlicks");

insert into grocery_items values("0012","tea");

insert into grocery_items values("0013","cofee");

insert into grocery_items values("0014","magie");

insert into grocery_items values("0015","chips");

insert into grocery_items values("0016","soup");

insert into grocery_items values("0017","detergent");

insert into grocery_items values("0018","harpic");

insert into grocery_items values("0019","loofah");

insert into grocery_items values("0020","facewash");

insert into grocery_items values("0021","facepack");

insert into grocery_items values("0022","kala hit");

insert into grocery_items values("0023","goodnight mosquito killer");

insert into grocery_items values("0024","dal");

insert into grocery_items values("0025","daadar");

insert into grocery_items values("0026","toothbursh");

insert into grocery_items values("0027","tooth brush");

insert into grocery_items values("0028","tooth paste");

insert into grocery_items values("0029","vicks");

insert into grocery_items values("0030","butter");

insert into grocery_items values("0031","Dettol");

insert into grocery_items values("0032","curd");

insert into grocery_items values("0033","butter");

insert into grocery_items values("0034","cheese");

insert into grocery_items values("0035","ghee");

insert into grocery_items values("0036","oil");

insert into grocery_items values("0037","rohafza");

insert into grocery_items values("0038","salted peanuts");

insert into grocery_items values("0039","dry fruits");

insert into grocery_items values("0040","paneer");

insert into grocery_items values("0041","buscuits");

insert into grocery_items values("0042","chocolate");

insert into grocery_items values("0043","chocolate syrup");

insert into grocery_items values("0044","baking soda");

insert into grocery_items values("0045","vanilla essence");

insert into grocery_items values("0046","milk");

insert into grocery_items values("0047","cookies");

insert into grocery_items values("0048","toast");

insert into grocery_items values("0049","papad");

insert into grocery_items values("0050","hing");



select * from grocery_items;
drop table grocery_items;
select * from grocery_items where code = '0001';
select * from grocery where username = "Mayank" and delivery_status= 'Not Delivered';