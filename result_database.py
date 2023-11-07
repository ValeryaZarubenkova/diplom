import sqlite3
import os

def f_database(artikul, total_count, positive_count, negative_count):
    database = 'database.db'
    #connection = sqlite3.connect(database)

    def check_db(filename):
        return os.path.exists(filename)

    if check_db(database):
        connection = sqlite3.connect(database)

    cursor = connection.cursor()

    # запрос на создание таблицы
    query1 = '''
    CREATE TABLE products(
        artikul integer primary key,
        total integer,
        positive integer,
        negative integer
    );'''
    query = "INSERT INTO products (artikul, total, positive, negative) VALUES (?, ?, ?, ?)"
    cursor.execute(query, (artikul, total_count, positive_count, negative_count))

    # выполняем 1-ый запрос
    #connection.execute(query1)
    # выполняем 2-ый запрос
    #connection.execute(query2)
    connection.commit()
    connection.close()
