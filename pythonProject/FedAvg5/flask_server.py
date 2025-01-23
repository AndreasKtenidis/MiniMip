#!/usr/bin/env python
# encoding: utf-8

from flask import Flask, request, jsonify
import sqlite3
import os

app = Flask(__name__)

DATABASE = "database.db"

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
        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS aggregation_step (
                    operation_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    client_id TEXT NOT NULL,                    
                    agg_func TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (operation_id,client_id, round, agg_func)
                    )
                ''')
        connection.commit()
        connection.close()
        print("Database initialized!")

# Automatically initialize the database when the app starts
initialize_database()

# "client_id": i, "agg_func": "sum", "value":5, "round":0, "agg_id":12431257
@app.route('/add-aggregation', methods=['POST'])
def add_aggregation():
    data = request.args
    operation_id =data.get('operation_id')
    client_id = data.get('client_id')
    exec_round = data.get('round')
    agg_func = data.get('agg_func')
    value = data.get('value')
    print("Adding Data from client: ",client_id,"round: ",exec_round,"agg_func: ",agg_func,"value: ",value,"")
    if not data or not client_id or not exec_round or not agg_func or not value :
        return jsonify({"error": "Filed is missing"}), 400
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute('INSERT INTO aggregation_step (operation_id,round,client_id,agg_func,value) VALUES (?,?,?,?,?)', (operation_id,exec_round,client_id,agg_func,value))
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
    operation_id = data.get('operation_id')
    exec_round = data.get('round')
    agg_func = data.get('agg_func')
    client_count:int= int(data.get('client_count'))

    connection = get_db_connection()
    cursor = connection.cursor()
    _sum=None
    _count=None
    print(client_count)
    if agg_func == 'sum' or agg_func == 'avg':
        cursor.execute('''SELECT sum(value)
                                FROM aggregation_step
                                WHERE operation_id=? AND round=? AND agg_func='sum'
                                HAVING count(value)=?''', (operation_id, exec_round, client_count))
        result = cursor.fetchone()
        if result:
            _sum = result[0]
    if agg_func == 'count' or agg_func == 'avg':
        cursor.execute('''SELECT sum(value)
                                        FROM aggregation_step
                                        WHERE operation_id=? AND round=? AND agg_func='count'
                                        HAVING count(value)=?''', (operation_id, exec_round, client_count))
        result = cursor.fetchone()
        if result:
            _count = result[0]
    connection.commit()
    connection.close()
    print("Count",_count)
    print("Sum",_sum)
    if agg_func == 'count' and _count is not None:
        return jsonify(_count), 200
    elif agg_func == 'avg' and _count is not None and _sum is not None:
        return jsonify(_sum/_count), 200
    elif agg_func == 'sum' and _sum is not None:
        return jsonify(_sum ), 200
    else:
        return jsonify(None), 300


# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)























#
#
#
#
# @app.route('/records', methods=['GET'])
# def get_records():
#     """
#     Reads all records or a specific record by name.
#     """
#     operation_id = request.args.get('operation_id')  # Query parameter to filter by name
#     round = request.args.get('round')  # Query parameter to filter by name
#     with open(DATA_FILE, 'r') as f:
#         aggregations = json.load(f)
#     if aggregations[operation_id]['client_count']== len(aggregations[operation_id][round]):
#         answer = 0
#         for record in aggregations[operation_id][round]:
#             answer += record['value']
#         return jsonify({'answer':answer}), 200
#     return jsonify({}), 200
#
#
# @app.route('/records', methods=['POST'])
# def add_record():
#     """
#     Adds a new record.
#     """
#     record = request.get_json()
#     with open(DATA_FILE, 'r') as f:
#         aggregations = json.load(f)
#
#     # data = {"client_id": "7435844", "agg_func": "sum", "value":5, "round":0, "operation_id":12431253}
#     operation_id = record.pop("operation_id", None)
#
#     if operation_id is None:
#         return jsonify({"error": "The operation_id and round of execution are not defined"}), 400
#     elif 'client_count' in record:
#         if operation_id not in aggregations:
#             aggregations[operation_id] = {}
#         aggregations[operation_id]['client_count']=record['client_count']
#     else:
#         round = record.pop("round", None)
#         if operation_id not in aggregations:
#             aggregations[operation_id] = {}
#             aggregations[operation_id][round] =[]
#         elif round not in aggregations[operation_id]:
#             aggregations[operation_id][round] =[]
#         # Check if the client id has already been examined for the round
#         if any(r['client_id'] == record['client_id'] for r in aggregations[operation_id][round]):
#             return jsonify({"error": "The client has already submitted it's value"}), 400
#         aggregations[operation_id][round].append(record)
#     with open(DATA_FILE, 'w') as f:
#         json.dump(aggregations, f, indent=2)
#     return jsonify({"message": "Record added", "record": record}), 201
#
#
# #
# # @app.route('/records', methods=['DELETE'])
# # def delete_record():
# #     """
# #     Deletes a record by name.
# #     """
# #     record = request.get_json()
# #     name = record.get('client_id')
# #     if not name:
# #         return jsonify({"error": "Name is required to delete a record"}), 400
# #
# #     with open(DATA_FILE, 'r') as f:
# #         records = json.load(f)
# #
# #     new_records = [r for r in records if r['client_id'] != name]
# #     if len(new_records) == len(records):
# #         return jsonify({"message": "Record not found"}), 404
# #
# #     with open(DATA_FILE, 'w') as f:
# #         json.dump(new_records, f, indent=2)
# #
# #     return jsonify({"message": "Record deleted", "name": name}), 200
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
