#!/usr/bin/env python
# encoding: utf-8

from flask import Flask, request, jsonify
import sqlite3
import os
from constants import Database,AGG
app = Flask(__name__)

DATABASE = "database.db"

table_creation= '''
                    CREATE TABLE IF NOT EXISTS aggregation_step (
                    operation_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    client_id TEXT NOT NULL,                    
                    agg_func TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (operation_id,client_id, round, agg_func)
                    )
                '''
local_aggregation = '''INSERT INTO aggregation_step (operation_id,round,client_id,agg_func,value) VALUES (?,?,?,?,?)'''
global_aggregation = '''SELECT sum(value)
                                FROM aggregation_step
                                WHERE operation_id=? AND round=? AND agg_func=?
                                HAVING count(value)=?'''

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
        connection.commit()
        connection.close()
        print("Database initialized!")

# Automatically initialize the database when the app starts
initialize_database()

def get_attribute(args, key:Database):
   return args.get(key.value)


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
    cursor.execute('SELECT * FROM aggregation_step')
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
        cursor.execute(global_aggregation, (operation_id, exec_round,AGG.SUM.value, client_count))
        result = cursor.fetchone()
        if result:
            _sum = result[0]
    if agg_func == 'count' or agg_func == AGG.AVG.value:
        cursor.execute(global_aggregation, (operation_id, exec_round,AGG.COUNT.value, client_count))
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





















