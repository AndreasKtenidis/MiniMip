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
        super().__init__(server,config,strategy,client_manager,server_fn)
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
        # Loop and wait until enough nodes are available.
        all_node_ids = driver.get_node_ids()

        node_ids=[]
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





        my_mapping={'x':'SepalLengthCm','y':'SepalWidthCm'}
        metadata={'driver':driver,'node_ids':node_ids}

        mx=self.agg_avg(metadata, 'x', 'iris',my_mapping)
        print(mx)
        # my = self.agg_avg( node_ids, 'y', 'iris',mapping)
        # mxy= self.agg_avg( node_ids, 'x*y', 'iris',mapping)
        # sx=math.sqrt(self.agg_avg( node_ids, 'x**2','iris',mapping)-mx**2)
        # sy = math.sqrt(
        #     self.agg_avg( node_ids, 'y**2', 'iris',mapping) - my ** 2)
        # return (mxy-mx*my)/(sx*sy)
        # sy = math.sqrt(self.aggSum(driver, node_ids, server_round, 'y^2') / count_ - my ^ 2)
        # self.aggSum(driver, node_ids, server_round, ["SepalLengthCm", "SepalWidthCm"])

    def get_results(self, metadata, agg_funcs: List[str], function, dataset: str, my_mapping: Dict[str, str]) -> \
            Dict[str, List[Number]]:
        driver = metadata['driver']
        node_ids = metadata['node_ids']
        server_round = 0
        recordset = RecordSet()
        configs = ConfigsRecord({"AGG_FUNC": agg_funcs,
                                 "COL_FUNC": [function],
                                 "DATASET": [dataset],
                                 "MAPPING": map_to_list(my_mapping)})
        recordset.configs_records["my_config"] = configs
        print("----->",recordset)

        messages = []
        for node_id in node_ids:  # one message for each node
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.QUERY,  # target `query` method in ClientApp
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(message)

        replies = driver.send_and_receive(messages)

        # Send messages and wait for all results

        # log(INFO, "Received %s/%s results", len(replies), len(messages))
        # answer = {"answer": 0}
        # for rep in replies:
        #     if rep.has_error():
        #         continue
        #     query_results = rep.content.metrics_records["query_results"]
        #     # Sum metrics
        #     for _, v in query_results.items():
        #         answer["answer"] += v
        #
        # return {}
        return {"count": [], "sum": []}

def map_to_list(my_mapping: Dict[str, str])->List[str]:
    out = []
    for key,value in my_mapping.items():
        out.append(key)
        out.append(value)
    return out



