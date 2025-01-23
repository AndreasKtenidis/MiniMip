import requests

class FlaskCommunicator:
    def __init__(self):
        pass


    @staticmethod
    def add_aggregation(operation_id, client_id, agg_round, agg_func, value):
        url = "http://localhost:5000/add-aggregation"
        response = requests.post(url, params={"operation_id": operation_id,"client_id":client_id,
                                              "round":agg_round, "agg_func":agg_func, "value":value})
        if response.status_code == 200:
            print("Success:", response.text)

    @staticmethod
    def get_aggregation(operation_id, exec_round, agg_func, client_count):
        url = "http://localhost:5000/get_aggregation"
        response = requests.get(url, params={"operation_id": operation_id,
                                             "round":exec_round, "agg_func":agg_func,
                                             "client_count":client_count})
        if response.status_code == 200:
            print("Success:", response.text)
        return response.text

