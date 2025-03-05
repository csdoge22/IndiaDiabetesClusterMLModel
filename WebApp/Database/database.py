import os
import sys
import mysql.connector
from mysql.connector.pooling import PooledMySQLConnection
from mysql.connector.abstracts import MySQLConnectionAbstract
from dotenv import load_dotenv
import numpy as np

load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
sys.path.insert(0,BASE_DIR)

from WebApp.app import retrieve_base_csv

def boolean_map():
    pass

def connect():
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')
    mydb = mysql.connector.connect(
        host="localhost",
        user=USERNAME,
        password=PASSWORD,
    )
    return mydb

def init_db():
    # connect to a MySQL session
    
    # print(mydb)
    mydb = connect()
    mycursor = mydb.cursor()
    mycursor.execute("CREATE DATABASE diabetes_db")
    mydb.commit()

def connect_db():
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')
    mydb = mysql.connector.connect(
        host="localhost",
        user=USERNAME,
        password=PASSWORD,
        database="diabetes_db"
    )
    return mydb

def init_stats():
    # create the tables
    mydb = connect_db()
    mycursor = mydb.cursor()
    sql = """CREATE TABLE people (
        id INT,
        username VARCHAR(255),
        age INT,
        gender VARCHAR(50),
        bmi DOUBLE,
        family_history BOOL,
        physical_activity VARCHAR(255),
        diet_type VARCHAR(255),
        smoking_status VARCHAR(50),
        alcohol_intake VARCHAR(50),
        stress_level VARCHAR(50),
        hypertension BOOL,
        cholesterol_level DOUBLE,
        fasting_blood_sugar DOUBLE,
        postprandial_blood_sugar DOUBLE,
        hba1c DOUBLE,
        heart_rate INT,
        waist_hip_ratio DOUBLE,
        urban_rural VARCHAR(50),
        health_insurance BOOL,
        regular_checkups BOOL,
        medication_for_chronic_conditions BOOL,
        pregnancies INT,
        polycystic_ovary_syndrome BOOL,
        glucose_tolerance_test_results DOUBLE,
        vitamin_d_level DOUBLE,
        c_protein_level DOUBLE,
        thyroid_condition BOOL,
        diabetes_status BOOL,
        PRIMARY KEY (id)
    )"""
    mycursor.execute(operation=sql)
    mydb.commit()
    # add the changes
    mydb.commit()

def add_stats():
    df = retrieve_base_csv()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(df)

    df.insert(0, column='username', value=np.array(['base']*df.shape[0]))
    df.insert(0, column='id', value=np.array(np.arange(0,df.shape[0])))
    
    mydb = connect_db()
    mycursor = mydb.cursor()

    for index, row in df.iterrows():
        sql = """INSERT INTO people
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        
        mycursor.execute(operation=sql, params=tuple(row.values))
        mydb.commit()
    print(df.iloc[0].values)
    pass

def init_users():
    mydb = connect_db()
    mycursor = mydb.cursor()
    sql = """CREATE TABLE users (
        username VARCHAR(25),
        password VARCHAR(50),
        PRIMARY KEY (username)
    )"""
    mycursor.execute(operation=sql)
    mydb.commit()
    # add the changes
    mydb.commit()

def drop_table(table_name):
    mydb = connect_db()
    mycursor = mydb.cursor()
    sql = f"DROP TABLE {table_name}"
    mycursor.execute(operation=sql)
    mydb.commit() 

drop_table('people')