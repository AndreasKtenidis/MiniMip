#!/usr/bin/env python
# encoding: utf-8

from flask import Flask, request, jsonify
import sqlite3
import os
from constants import Database,AGG
app = Flask(__name__)

DATABASE = "database.db"

opp_table_creation = ''' CREATE TABLE operations (
                         operation_id INTEGER PRIMARY KEY AUTOINCREMENT,                        
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                         client_count INTEGER
                        )
'''
opp_add = 'INSERT INTO operations (client_count) VALUES (?)'
table_creation= ''' CREATE TABLE IF NOT EXISTS aggregation_step (
                    operation_id INTEGER NOT NULL,
                    round INTEGER NOT NULL,
                    client_id INTEGER NOT NULL,                    
                    agg_func TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (operation_id,client_id, round, agg_func)
                    )
                '''
select_all_operations = '''SELECT * FROM operations'''

local_aggregation = '''INSERT INTO aggregation_step (operation_id,round,client_id,agg_func,value) VALUES (?,?,?,?,?)'''
global_count = '''SELECT sum(value)
                                FROM aggregation_step
                                WHERE operation_id=? AND round=? AND agg_func='COUNT'
                                HAVING count(value)=?'''
global_sum = '''SELECT sum(value)
                                FROM aggregation_step
                                WHERE operation_id=? AND round=? AND agg_func='SUM'
                                HAVING count(value)=?'''
select_all = '''SELECT * FROM aggregation_step'''

# Function to get a database connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # To return rows as dictionaries
    return conn

# Function to initialize the database
def initialize_database():
    if not os.path.exists(DATABASE):  # Check if the database file exists
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(table_creation)
        cursor.execute(opp_table_creation)
        connection.commit()
        connection.close()
        print("Database initialized!")

# Automatically initialize the database when the app starts
initialize_database()

def get_attribute(args, key:Database):
   return args.get(key.value)

@app.route('/add_operation', methods=['GET'])
def add_operation():
    data = request.args
    client_count=get_attribute(data, Database.CLIENT_COUNT)
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(opp_add,client_count)
    new_operation_id = cursor.lastrowid
    connection.commit()
    cursor.close()
    connection.close()
    return jsonify(new_operation_id), 200


@app.route('/get_operations', methods=['GET'])
def get_operations():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(select_all_operations)
    users = cursor.fetchall()
    connection.commit()
    connection.close()
    return jsonify([dict(row) for row in users]), 200

@app.route('/add-aggregation', methods=['POST'])
def add_aggregation():
    data = request.args
    operation_id =get_attribute(data, Database.OPP)
    client_id = get_attribute(data, Database.CLIENT)
    exec_round = get_attribute(data, Database.ROUND)
    agg_func = get_attribute(data, Database.AGG)
    value = get_attribute(data, Database.VALUE)
    print("Adding Data from client: ",client_id,"round: ",exec_round,"agg_func: ",agg_func,"value: ",value,"")
    if not data or not client_id or not exec_round or not agg_func or not value :
        return jsonify({"error": "Filed is missing"}), 400
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(local_aggregation, (operation_id, exec_round, client_id, agg_func, value))
    connection.commit()
    connection.close()
    return jsonify({"message": "User added successfully!"}), 201

@app.route('/get-aggregations', methods=['GET'])
def get_aggregations():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(select_all)
    users = cursor.fetchall()
    connection.commit()
    connection.close()
    return jsonify([dict(row) for row in users]), 200

@app.route('/get_aggregation', methods=['GET'])
def get_aggregation():
    data = request.args
    operation_id = get_attribute(data, Database.OPP)
    exec_round = get_attribute(data, Database.ROUND)
    agg_func = get_attribute(data, Database.AGG)
    client_count:int= int(get_attribute(data, Database.CLIENT_COUNT))

    connection = get_db_connection()
    cursor = connection.cursor()
    _sum=None
    _count=None
    print(client_count)


    if agg_func == AGG.SUM.value or agg_func == AGG.AVG.value:
        cursor.execute(global_sum, (operation_id, exec_round, client_count))

        result = cursor.fetchone()
        if result:
            _sum = result[0]
    if agg_func == AGG.COUNT.value or agg_func == AGG.AVG.value:
        print("==>", agg_func, AGG.COUNT.value)
        cursor.execute(global_count, (operation_id, exec_round, client_count))
        print(global_count)
        print(operation_id, exec_round, client_count)
        result = cursor.fetchone()
        if result:
            _count = result[0]
    connection.commit()
    connection.close()
    print("Count",_count)
    print("Sum",_sum)
    if agg_func == AGG.COUNT.value and _count is not None:
        return jsonify(_count), 200
    elif agg_func == AGG.AVG.value and _count is not None and _sum is not None:
        return jsonify(_sum/_count), 200
    elif agg_func == AGG.SUM.value and _sum is not None:
        return jsonify(_sum ), 200
    else:
        return jsonify(None), 300


# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
