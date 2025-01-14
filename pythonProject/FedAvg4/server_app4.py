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



def map_to_list(my_mapping: Dict[str, str])->List[str]:
    out = []
    for key,value in my_mapping.items():
        out.append(key)
        out.append(value)
    return out



class AGG:
    AVG = "AVG"
    SUM = "SUM"
    COUNT = "COUNT"

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


        mx=self.aggAvg(driver, node_ids, server_round,'x', ["SepalLengthCm", "SepalWidthCm"])
        my = self.aggAvg(driver, node_ids, server_round, 'y', ["SepalLengthCm", "SepalWidthCm"])
        mxy= self.aggAvg(driver, node_ids, server_round, 'x*y', ["SepalLengthCm", "SepalWidthCm"])
        sx=math.sqrt(self.aggAvg(driver, node_ids, server_round, 'x**2',["SepalLengthCm", "SepalWidthCm"])-mx**2)
        sy = math.sqrt(
            self.aggAvg(driver, node_ids, server_round, 'y**2', ["SepalLengthCm", "SepalWidthCm"]) - my ** 2)
        print((mxy-mx*my)/(sx*sy))
        return (mxy-mx*my)/(sx*sy)
        # sy = math.sqrt(self.aggSum(driver, node_ids, server_round, 'y^2') / count_ - my ^ 2)
        # self.aggSum(driver, node_ids, server_round, ["SepalLengthCm", "SepalWidthCm"])

    def aggAvg(self,driver: Driver, node_ids, server_round,function:str, features):
        recordset = RecordSet()


        my_mapping = {'x': 'SepalLengthCm', 'y': 'SepalWidthCm'}
        configs = ConfigsRecord({"AGG_FUNC": AGG.AVG,
                                 "MAPPING":map_to_list(my_mapping),
                                 "COL_FUNC":function
                                 })
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
        answer = {AGG.SUM:0, AGG.COUNT:0}
        for rep in replies:
            if rep.has_error():
                continue
            query_results = rep.content.metrics_records["query_results"]
            # Sum metrics
            for k,v in query_results.items():
                answer[k] += v
        print("!!!!!!!!!!!!!!!",answer)
        # return answer["answer"]
        return answer[AGG.SUM]/answer[AGG.COUNT]

app = MyServerApp()