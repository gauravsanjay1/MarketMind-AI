import operator
from typing import Annotated, Any, TypedDict
from pydantic import BaseModel
from ..schemas.common import ProjectBrief, TaskStatus

AGENT_ORDER = ["competitor", "market_fit", "marketing", "influencer"]
DEPENDENCIES = {
    "competitor": [],
    "market_fit": [],
    "marketing": ["competitor", "market_fit"],
    "influencer": ["marketing"],
}


class LogEntry(BaseModel):
    agent: str
    event: str
    detail: str = ""


class WorkflowState(TypedDict, total=False):
    brief: ProjectBrief
    knowledge: str                       # long-term memory + retrieved document chunks
    plan: list[str]                      # agents selected by the orchestrator
    results: dict[str, Any]              # agent -> report (pydantic)
    sources: dict[str, list]             # agent -> sources used
    status: dict[str, TaskStatus]
    attempts: dict[str, int]
    errors: dict[str, str]
    logs: Annotated[list[LogEntry], operator.add]
    missing: list[str]
    final: dict[str, Any]
