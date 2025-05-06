import mysql.connector
import os

def connection_db():
    conn = mysql.connector.connect(user='root', password="Advika@2018", host='localhost')
    cursor = conn.cursor()
    cursor.execute("use user_data;")
    return conn, cursor