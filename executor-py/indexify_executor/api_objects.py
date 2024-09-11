from pydantic import BaseModel
from typing import Dict, Any

class Task(BaseModel):
    id: str
    namespace: str
    compute_graph: str
    compute_fn: str
    invocation_id: str
    input_id: str

class ExecutorMetadata(BaseModel):
    id: str
    address: str
    runner_name: str
    labels: Dict[str, Any]
