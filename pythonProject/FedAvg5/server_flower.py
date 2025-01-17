"""pandas_example: A Flower / Pandas app."""

import random
import time

from logging import INFO
from typing import Optional,Dict,List
import math

from flwr.common import Context, MessageType, RecordSet, Message, ParametersRecord, ConfigsRecord
from flwr.common.logger import log
from flwr.server import Driver, ServerApp

from flwr. server. server import Server
from flwr. server. server_config import ServerConfig
from flwr.server.strategy import Strategy
from flwr. server.client_manager import ClientManager
from flwr. server.typing import ServerFn

from abstractions import Complex_Operation
from server_abstract import AGG, PARAMS,Executor

def map_to_list(my_mapping: Dict[str, str])->List[str]:
    out = []
    for key,value in my_mapping.items():
        out.append(key)
        out.append(value)
    return out

class MyServerApp(ServerApp):
    """A custom application that extends ServerApp."""

    def __init__(
            self,
            server: Optional[Server] = None,
            config: Optional[ServerConfig] = None,
            strategy: Optional[Strategy] = None,
            client_manager: Optional[ClientManager] = None,
            server_fn: Optional[ServerFn] = None,
    ):
        super().__init__(server,config,strategy,client_manager,server_fn)

        @self.main()
        def main(driver: Driver, context: Context) -> None:
            self.my_main(driver,context)

    def my_main(self, driver: Driver, context: Context) -> None:

        min_nodes = 2
        num_rounds = 2
        min_nodes = 10
        fraction_sample = 1

        server_round = 0
        log(INFO, "")  # Add newline for log readability
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

        # Loop and wait until enough nodes are available.
        all_node_ids = []
        while len(all_node_ids) < min_nodes:
            all_node_ids = driver.get_node_ids()
            if len(all_node_ids) >= min_nodes:
                # Sample nodes
                num_to_sample = int(len(all_node_ids) * fraction_sample)
                node_ids = random.sample(all_node_ids, num_to_sample)
                break
            log(INFO, "Waiting for nodes to connect...")
            time.sleep(2)

        log(INFO, "Sampled %s nodes (out of %s)", len(node_ids), len(all_node_ids))

        my_mapping = {'x': 'SepalLengthCm', 'y': 'SepalWidthCm'}

        executor = FlowerExecutor(driver, node_ids, server_round,my_mapping)
        print("!!!!!!!!!!!", Complex_Operation(executor).value())





class FlowerExecutor(Executor):
    def __init__(self, driver, node_ids, server_round,mapping:Dict[str,str]):
        self.driver = driver
        self.node_ids = node_ids
        self.server_round = server_round
        self.mapping = mapping

    def AVG(self, function:str):
        recordset = RecordSet()

        configs = ConfigsRecord({PARAMS.AGG_FUNC.__str__() : AGG.AVG.__str__(),
                                 PARAMS.MAPPING.__str__():map_to_list(self.mapping),
                                 PARAMS.COL_FUNC.__str__():function
                                 })
        recordset.configs_records["my_config"] = configs

        print(recordset)
        messages = []
        for node_id in self.node_ids:  # one message for each node
            message = self.driver.create_message(
                content=recordset,
                message_type=MessageType.QUERY,  # target `query` method in ClientApp
                dst_node_id=node_id,
                group_id=str(self.server_round),
            )
            messages.append(message)

        # Send messages and wait for all results
        replies = self.driver.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))
        answer = {AGG.SUM.__str__():0, AGG.COUNT.__str__():0}
        for rep in replies:
            if rep.has_error():
                continue
            query_results = rep.content.metrics_records[PARAMS.RESULTS.__str__()]
            # Sum metrics
            for k,v in query_results.items():
                answer[k] += v
        return answer[AGG.SUM.__str__()]/answer[AGG.COUNT.__str__()]

app = MyServerApp()