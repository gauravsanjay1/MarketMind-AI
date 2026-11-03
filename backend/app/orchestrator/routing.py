from ..schemas.common import TaskStatus
from .state import AGENT_ORDER, DEPENDENCIES, WorkflowState


def next_agent(state: WorkflowState) -> str:
    """Pick next runnable agent from plan; END-ish 'validate' when none left."""
    status = state.get("status", {})
    for a in AGENT_ORDER:
        if a not in state.get("plan", []):
            continue
        if status.get(a) in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED):
            continue
        deps = DEPENDENCIES[a]
        if any(status.get(d) == TaskStatus.FAILED for d in deps if d in state["plan"]):
            continue  # handled by skip node
        if all(status.get(d) == TaskStatus.DONE for d in deps if d in state["plan"]):
            return a
    return "validate"


def route(state: WorkflowState) -> str:
    # blocked agents (failed upstream) get skipped before choosing
    status = state.get("status", {})
    for a in state.get("plan", []):
        if status.get(a) in (TaskStatus.PENDING, None) and any(
            status.get(d) == TaskStatus.FAILED for d in DEPENDENCIES[a] if d in state["plan"]
        ):
            return "skip_blocked"
    return next_agent(state)
