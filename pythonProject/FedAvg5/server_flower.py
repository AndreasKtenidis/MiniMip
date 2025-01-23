"""pandas_example: A Flower / Pandas app."""

import random
import time

from logging import INFO
from typing import Optional,Dict,List


from flwr.common import Context, MessageType, RecordSet,  ConfigsRecord
from flwr.common.logger import log
from flwr.server import Driver, ServerApp

from flwr. server. server import Server
from flwr. server. server_config import ServerConfig
from flwr.server.strategy import Strategy
from flwr. server.client_manager import ClientManager
from flwr. server.typing import ServerFn

from server_abstract import AGG, PARAMS

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
        print(driver.get_node_ids())
        num_rounds = 2
        min_nodes = 5
        fraction_sample = 1

        server_round = 0

        # Loop and wait until enough nodes are available.

        node_ids, all_node_ids = self.get_available_nodes(driver,min_nodes,fraction_sample)

        log(INFO, "Sampled %s nodes (out of %s)", len(node_ids), len(all_node_ids))

        my_mapping = {'x': 'SepalLengthCm', 'y': 'SepalWidthCm'}

        recordset = RecordSet()

        configs = ConfigsRecord({PARAMS.AGG_FUNC.__str__(): AGG.AVG.__str__(),
                                 PARAMS.MAPPING.__str__(): map_to_list(my_mapping),
                                 PARAMS.COL_FUNC.__str__(): "x**2"
                                 })
        recordset.configs_records["Statics_Start"] = configs

        print(recordset)
        messages = []
        for node_id in node_ids:  # one message for each node
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.QUERY,  # target `query` method in ClientApp
                dst_node_id=node_id,
                group_id=str(0),
            )
            messages.append(message)

        driver.send_and_receive(messages)

    @staticmethod
    def get_available_nodes(driver, min_nodes, fraction_sample):
        all_node_ids = []
        node_ids=[]
        while len(all_node_ids) < min_nodes:
            all_node_ids = driver.get_node_ids()
            if len(all_node_ids) >= min_nodes:
                # Sample nodes
                num_to_sample = int(len(all_node_ids) * fraction_sample)
                node_ids = random.sample(all_node_ids, num_to_sample)
                break
            time.sleep(2)
        return node_ids,all_node_ids



app = MyServerApp()