from collections import defaultdict
from queue import deque
from typing import Any, Dict, List, Optional, Type, Union

import cbor2
from nanoid import generate
from pydantic import BaseModel, Json
from rich import print

from indexify.base_client import IndexifyClient 
from indexify.functions_sdk.data_objects import BaseData, File, RouterOutput
from indexify.functions_sdk.graph import Graph
from indexify.functions_sdk.local_cache import CacheAwareFunctionWrapper


# Holds the outputs of a
class ContentTree(BaseModel):
    id: str
    outputs: Dict[str, List[BaseData]]


class LocalRunner(IndexifyClient):
    def __init__(self, cache_dir: str = "./indexify_local_runner_cache"):
        self._cache_dir = cache_dir
        self._graphs: Dict[str, Graph] = {}
        self._results: Dict[str, Dict[str, List[BaseData]]] = {}
        self._cache = CacheAwareFunctionWrapper(self._cache_dir)

    def register_extraction_graph(self, graph: Graph):
        self._graphs[graph.name] = graph

    def run_from_serialized_code(self, code: bytes, **kwargs):
        g = Graph.deserialize(graph=code)
        self.run(g, **kwargs)

    def run(self, g: Graph, **kwargs):
        input = cbor2.dumps(kwargs)
        print(f"[bold] Invoking {g._start_node}[/bold]")
        outputs = defaultdict(list)
        content_id = generate()
        self._results[content_id] = outputs
        self._run(g, input, outputs)
        return content_id

    def _run(
        self,
        g: Graph,
        initial_input: bytes,
        outputs: Dict[str, List[bytes]],
    ):
        queue = deque([(g._start_node.name, initial_input)])
        while queue:
            node_name, input_bytes = queue.popleft()
            cached_output_bytes: Optional[List[bytes]] = self._cache.get(
                g.name, node_name, input_bytes
            )
            if cached_output_bytes is not None:
                for cached_output in cached_output_bytes:
                    outputs[node_name].append(cached_output)
            else:
                function_results: List[bytes] = g.invoke_fn_ser(node_name, input_bytes)
                outputs[node_name].extend(function_results)
                self._cache.set(
                    g.name,
                    node_name,
                    input_bytes,
                    function_results,
                )

            function_outputs = outputs[node_name]

            out_edges = g.edges.get(node_name, [])
            # Figure out if there are any routers for this node
            for i, edge in enumerate(out_edges):
                if edge in g.routers:
                    out_edges.remove(edge)
                    for output in function_outputs:
                        dynamic_edges = self._route(g, edge, output) or []
                        for dynamic_edge in dynamic_edges.edges:
                            if dynamic_edge in g.nodes:
                                print(
                                    f"[bold]dynamic router returned node: {dynamic_edge}[/bold]"
                                )
                                out_edges.append(dynamic_edge)
            for out_edge in out_edges:
                print(
                    f"invoking {out_edge} with {len(function_outputs)} outputs from {node_name}"
                )
                for output in function_outputs:
                    queue.append((out_edge, output))

    def _route(self, g: Graph, node_name: str, input: bytes) -> Optional[RouterOutput]:
        return g.invoke_router(node_name, input)

    def register_graph(self, graph: Graph):
        self._graphs[graph.name] = graph

    def graphs(self) -> str:
        return list(self._graphs.keys())

    def namespaces(self) -> str:
        return "local"

    def create_namespace(self, namespace: str):
        pass

    def invoke_graph_with_object(self, graph: str, **kwargs) -> str:
        graph = self._graphs[graph]
        for key, value in kwargs.items():
            if isinstance(value, BaseModel):
                kwargs[key] = value.model_dump()

        return self.run(graph, **kwargs)

    def invoke_graph_with_file(
        self, graph: str, path: str, metadata: Optional[Dict[str, Json]] = None
    ) -> str:
        graph = self._graphs[graph]
        with open(path, "rb") as f:
            data = f.read()
            file = File(data, metadata=metadata)
        return self.run(graph, file=file)

    def graph_outputs(
        self,
        graph: str,
        ingested_object_id: str,
        extractor_name: str,
        block_until_done: bool = True,
    ) -> Union[Dict[str, List[Any]], List[Any]]:
        if ingested_object_id not in self._results:
            raise ValueError(
                f"No results found for ingested object {ingested_object_id}"
            )
        if extractor_name not in self._results[ingested_object_id]:
            raise ValueError(
                f"No results found for extractor {extractor_name} on ingested object {ingested_object_id}"
            )
        results = []
        fn_model = self._graphs[graph].get_function(extractor_name).get_output_model()
        for result in self._results[ingested_object_id][extractor_name]:
            output_dict = cbor2.loads(result)
            payload = fn_model.model_validate(output_dict["payload"])
            results.append(payload)
        return results