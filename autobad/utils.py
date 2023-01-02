from typing import Dict, Optional

import graphviz

from autobad.graph import Graph


def graph_viz(node_names: Optional[Dict[int, str]] = {}):
    """
    Use graphviz to visualise the backwards graph, node_names is a map from id
    to a string identifier. It isn't necessary but will make the graph more
    understandable.
    """
    dot = graphviz.Digraph()
    for (node, children) in Graph.get_instance()._graph.items():
        parent_name = node_names.get(node, str(node))
        dot.node(parent_name)
        for _, c in children:
            child_name = str(node_names.get(id(c), id(c)))
            dot.node(child_name)
            dot.edge(parent_name, child_name)
    dot.render(view=True)
