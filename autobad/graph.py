from collections import defaultdict
from typing import Dict, List, Tuple

import autobad as ab
from autobad.typing import Vjp


class Graph:
    __instance = None

    @staticmethod
    def get_instance():
        if Graph.__instance == None:
            Graph()
        return Graph.__instance

    def __init__(self):
        if Graph.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Graph.__instance = self
        self._graph: Dict[ab.Tensor, List[Tuple[Vjp, ab.Tensor]]] = defaultdict(list)

    @staticmethod
    def clear() -> None:
        Graph.get_instance()._graph = defaultdict(list)

    @staticmethod
    def add(parent: ab.Tensor, children: List[ab.Tensor]) -> None:
        Graph.get_instance()._graph[id(parent)].extend(children)

    @staticmethod
    def get(node: ab.Tensor) -> List[ab.Tensor]:
        return Graph.get_instance()._graph[id(node)]
