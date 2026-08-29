"""Executes a LiteGraph-serialized graph server-side.

LiteGraph's graph.serialize() shape (the parts we care about):
{
  "nodes": [
    {"id": 1, "type": "cv/LoadImage", "inputs": [...], "outputs": [...],
     "widgets_values": [...]},
    ...
  ],
  "links": [
    [link_id, origin_node_id, origin_slot, target_node_id, target_slot, type],
    ...
  ]
}

We rebuild a dependency graph from the links, topologically sort the nodes,
then run each node's `run()` with keyword arguments assembled from its
connected inputs (numpy arrays produced by upstream nodes) and its widget
values (raw values from the UI).
"""

import time
from collections import defaultdict, deque

from .registry import REGISTRY


class GraphExecutionError(Exception):
    def __init__(self, node_id, message):
        super().__init__(message)
        self.node_id = node_id
        self.message = message


def _topological_order(nodes_by_id, links):
    in_degree = {nid: 0 for nid in nodes_by_id}
    dependents = defaultdict(list)  # origin_id -> [target_id, ...]

    for link in links:
        _link_id, origin_id, _origin_slot, target_id, _target_slot, _type = link
        if origin_id not in nodes_by_id or target_id not in nodes_by_id:
            continue
        dependents[origin_id].append(target_id)
        in_degree[target_id] += 1

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    order = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for dep in dependents[nid]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(order) != len(nodes_by_id):
        raise GraphExecutionError(None, "Graph contains a cycle - cannot execute")

    return order


def execute_graph(graph_data: dict) -> dict:
    """Runs the graph. Returns {node_id: {output_name: value}} for every
    node flagged IS_OUTPUT_NODE, where image values are still raw numpy
    arrays (the caller is responsible for encoding them for the response)."""

    raw_nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])
    nodes_by_id = {n["id"]: n for n in raw_nodes}

    links_by_id = {}
    for link in links:
        links_by_id[link[0]] = link

    order = _topological_order(nodes_by_id, links)

    node_outputs = {}  # node_id -> [value_slot0, value_slot1, ...]
    output_results = {}
    graph_start = time.perf_counter()

    for node_id in order:
        node = nodes_by_id[node_id]
        node_type = node["type"]
        cls = REGISTRY.get(node_type)
        if cls is None:
            raise GraphExecutionError(node_id, f"Unknown node type '{node_type}'")

        kwargs = {}

        for slot_idx, inp in enumerate(node.get("inputs") or []):
            link_id = inp.get("link")
            if link_id is None:
                continue
            link = links_by_id.get(link_id)
            if link is None:
                continue
            _lid, origin_id, origin_slot, _tid, _tslot, _type = link
            try:
                value = node_outputs[origin_id][origin_slot]
            except (KeyError, IndexError):
                raise GraphExecutionError(
                    node_id, f"Missing upstream value for input '{inp.get('name')}'"
                )
            kwargs[inp["name"]] = value

        widget_values = node.get("widgets_values") or []
        for i, widget_def in enumerate(cls.WIDGETS):
            if i < len(widget_values):
                kwargs[widget_def["name"]] = widget_values[i]
            elif "default" in widget_def:
                kwargs[widget_def["name"]] = widget_def["default"]

        node_start = time.perf_counter()
        try:
            instance = cls()
            result = instance.run(**kwargs) or {}
        except Exception as exc:  # noqa: BLE001 - surface any node failure to the UI
            raise GraphExecutionError(node_id, f"{cls.NAME}: {exc}") from exc
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        print(f"[cvnodes] node {node_id} ({cls.NAME}) took {elapsed_ms:.1f} ms")

        node_outputs[node_id] = [result.get(o["name"]) for o in cls.OUTPUTS]

        if cls.IS_OUTPUT_NODE:
            output_results[node_id] = {
                "type": node_type,
                "name": cls.NAME,
                "values": result,
            }

    total_ms = (time.perf_counter() - graph_start) * 1000
    print(f"[cvnodes] graph finished: {len(order)} node(s) in {total_ms:.1f} ms")

    return output_results
