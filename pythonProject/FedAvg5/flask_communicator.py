import requests
from constants import Database,AGG

class FlaskCommunicator:
    def __init__(self):
        pass

    @staticmethod
    def get_operation(client_count):
        url = "http://localhost:5000/add_operation"
        response = requests.get(url, params={Database.CLIENT_COUNT.value:client_count})
        if response.status_code == 200:
            print("Success:", response.text)
        return int(response.text)

    @staticmethod
    def add_aggregation(operation_id, client_id, agg_round, agg_func, value):
        url = "http://localhost:5000/add-aggregation"
        response = requests.post(url, params={Database.OPP.value: operation_id,
                                              Database.CLIENT.value:client_id,
                                              Database.ROUND.value:agg_round,
                                              Database.AGG.value:agg_func,
                                              Database.VALUE.value:value})
        if response.status_code == 200:
            print("Success:", response.text)

    @staticmethod
    def get_aggregation(operation_id, exec_round, agg_func, client_count):
        url = "http://localhost:5000/get_aggregation"
        response = requests.get(url, params={Database.OPP.value: operation_id,
                                             Database.ROUND.value:exec_round,
                                             Database.AGG.value:agg_func,
                                             Database.CLIENT_COUNT.value:client_count})
        if response.status_code == 200:
            print("Success:", response.text)
        return response.text

