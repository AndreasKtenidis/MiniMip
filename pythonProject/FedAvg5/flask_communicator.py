import requests

class Flask_Communicator:
    def __init__(self):
        pass

    def add_operation(self, operation_id, clients):
        url = "http://localhost:5000/add-operation"
        response = requests.post(url, params={"operation_id": operation_id, "clients": clients})
        if response.status_code == 200:
            print("Success:", response.text)

    def add_aggregation(self, operation_id,     client_id ,    round,    agg_func,    value):
        url = "http://localhost:5000/add-aggregation"
        response = requests.post(url, params={"operation_id": operation_id,"client_id":client_id,
                                              "round":round,"agg_func":agg_func,"value":value})
        if response.status_code == 200:
            print("Success:", response.text)

    def get_aggregation(self, operation_id,round , agg_func):
        url = "http://localhost:5000/get_aggregation"
        response = requests.get(url, params={"operation_id": operation_id,"round":round,"agg_func":agg_func})
        if response.status_code == 200:
            print("Success:", response.text)

# Define the URL with query parameters


# Parameters to be included in the query string
params = {
    "operation_id": 1,
    "clients": 4
}

commune = Flask_Communicator()
# commune.add_operation(3,2)
# commune.add_aggregation(3,"asdfasdf",1,"sum",23)
# commune.add_aggregation(3,"asdf2asdf",1,"sum",78)
# commune.add_aggregation(3,"asdfasdf",1,"count",7)
# commune.add_aggregation(3,"asdf2asdf",1,"count",4)

a=commune.get_aggregation(3,1,"avg")
print(a)