"""pandas_example: A Flower / Pandas app."""

import random
import time
from logging import INFO
from numbers import Number
from typing import Optional, List, Dict
import math

from flwr.common import Context, MessageType, RecordSet, Message, ParametersRecord, ConfigsRecord
from flwr.common.logger import log
from flwr.server import Driver, ServerApp

from flwr. server. server import Server
from flwr. server. server_config import ServerConfig
from flwr.server.strategy import Strategy
from flwr. server.client_manager import ClientManager
from flwr. server.typing import ServerFn

from pythonProject.FedAvg5.interface.ServerInterface import ClientInteface


class MyServerApp(ServerApp,ClientInteface):
    """A custom application that extends ServerApp."""



    def __init__(
            self,
            server: Optional[Server] = None,
            config: Optional[ServerConfig] = None,
            strategy: Optional[Strategy] = None,
            client_manager: Optional[ClientManager] = None,
            server_fn: Optional[ServerFn] = None,
    ):
        super().__init__(self,server,config,strategy,client_manager,server_fn)
        @self.main()
        def main(driver: Driver, context: Context) -> None:
            self.my_main(driver,context)

    def my_main(self, driver: Driver, context: Context) -> None:
        """This `ServerApp` construct a histogram from partial-histograms reported by the
        `ClientApp`s."""
        # num_rounds = context.run_config["num-server-rounds"]
        min_nodes = 2
        # fraction_sample = context.run_config["fraction-sample"]
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

        count_ = self.aggCount(driver, node_ids, server_round)
        mx=self.aggSum(driver, node_ids, server_round,'x', ["SepalLengthCm", "SepalWidthCm"])/count_
        my = self.aggSum(driver, node_ids, server_round, 'y', ["SepalLengthCm", "SepalWidthCm"])/count_
        mxy= self.aggSum(driver, node_ids, server_round, 'x*y', ["SepalLengthCm", "SepalWidthCm"])/count_
        sx=math.sqrt(self.aggSum(driver, node_ids, server_round, 'x**2',["SepalLengthCm", "SepalWidthCm"])/count_-mx**2)
        sy = math.sqrt(
            self.aggSum(driver, node_ids, server_round, 'y**2', ["SepalLengthCm", "SepalWidthCm"]) / count_ - my ** 2)
        return (mxy-mx*my)/(sx*sy)
        # sy = math.sqrt(self.aggSum(driver, node_ids, server_round, 'y^2') / count_ - my ^ 2)
        # self.aggSum(driver, node_ids, server_round, ["SepalLengthCm", "SepalWidthCm"])

    def aggSum(self,driver: Driver, node_ids, server_round,function:str, features):
        recordset = RecordSet()

        configs = ConfigsRecord({"AGG_SUM": [function]+features})
        recordset.configs_records["my_config"] = configs
        print(recordset)
        messages = []
        for node_id in node_ids:  # one message for each node
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.QUERY,  # target `query` method in ClientApp
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(message)

        # Send messages and wait for all results
        replies = driver.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))
        answer = {"answer":0}
        for rep in replies:
            if rep.has_error():
                continue
            query_results = rep.content.metrics_records["query_results"]
            # Sum metrics
            for _,v in query_results.items():
                answer["answer"] += v
        return answer["answer"]

    def aggCount(self,driver: Driver, node_ids, server_round):
        recordset = RecordSet()
        print("--->",node_ids)

        configs = ConfigsRecord({"AGG_COUNT": ["*"]})
        recordset.configs_records["my_config"] = configs
        print(recordset)
        messages = []
        for node_id in node_ids:  # one message for each node
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.QUERY,  # target `query` method in ClientApp
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(message)

        # Send messages and wait for all results
        replies = driver.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))
        answer = {"answer":0}
        for rep in replies:
            if rep.has_error():
                continue
            query_results = rep.content.metrics_records["query_results"]
            # Sum metrics
            for _,v in query_results.items():
                answer["answer"] += v
        return answer["answer"]

    def get_results(self, agg_funcs: List[str], node_ids: List[int], function, dataset: str, mapping: Dict[str:str]) -> \
            Dict[str, List[Number]]:

        pass