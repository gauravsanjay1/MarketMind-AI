import logging
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from ..agents.base import BaseAgent
from ..agents.competitor_agent import CompetitorAgent
from ..agents.market_fit_agent import MarketFitAgent
from ..agents.marketing_agent import MarketingAgent
from ..agents.influencer_agent import InfluencerAgent
from ..schemas.common import ProjectBrief, TaskStatus
from ..tools.llm import LLMClient
from ..tools.web_search import SearchProvider
from .routing import route
from .state import AGENT_ORDER, DEPENDENCIES, LogEntry, WorkflowState

log = logging.getLogger("marketmind")


class Plan(BaseModel):
    agents: list[str]
    rationale: str = ""


def build_graph(llm: LLMClient, search: SearchProvider, max_retries: int = 2):
    agents: dict[str, BaseAgent] = {
        "competitor": CompetitorAgent(llm, search),
        "market_fit": MarketFitAgent(llm, search),
        "marketing": MarketingAgent(llm, search),
        "influencer": InfluencerAgent(llm, search),
    }

    def plan_node(state: WorkflowState):
        brief = state["brief"]
        p = llm.structured(
            "Select which of these agents are needed for the objective: " + ", ".join(AGENT_ORDER)
            + ". Return only names from that list.",
            f"Objective: {brief.objective}\nProduct: {brief.product_description}", Plan)
        wanted = {a for a in AGENT_ORDER if a in p.agents}
        stack = list(wanted)
        while stack:  # transitive closure of upstream dependencies
            for d in DEPENDENCIES[stack.pop()]:
                if d not in wanted:
                    wanted.add(d)
                    stack.append(d)
        chosen = [a for a in AGENT_ORDER if a in wanted]
        return {"plan": chosen, "results": {}, "sources": {}, "errors": {}, "attempts": {},
                "status": {a: TaskStatus.PENDING for a in chosen},
                "logs": [LogEntry(agent="orchestrator", event="plan", detail=p.rationale or ",".join(chosen))]}

    def make_node(name: str):
        def node(state: WorkflowState):
            brief: ProjectBrief = state["brief"]
            context = {}
            if state.get("knowledge"):
                context["company_knowledge"] = state["knowledge"]   # first so truncation keeps it
            context |= {k: v.model_dump() for k, v in state["results"].items()}
            status, results = dict(state["status"]), dict(state["results"])
            attempts, errors, srcs = dict(state["attempts"]), dict(state["errors"]), dict(state["sources"])
            logs = []
            for attempt in range(1, max_retries + 2):
                attempts[name] = attempt
                logs.append(LogEntry(agent=name, event="start", detail=f"attempt {attempt}"))
                try:
                    report, ev = agents[name].run(brief, context)
                    results[name], srcs[name] = report, ev.sources
                    status[name] = TaskStatus.DONE
                    logs.append(LogEntry(agent=name, event="done"))
                    break
                except Exception as e:
                    errors[name] = f"{type(e).__name__}: {e}"
                    logs.append(LogEntry(agent=name, event="error", detail=errors[name]))
                    status[name] = TaskStatus.FAILED
            return {"status": status, "results": results, "attempts": attempts,
                    "errors": errors, "sources": srcs, "logs": logs}
        return node

    def skip_blocked(state: WorkflowState):
        status, logs = dict(state["status"]), []
        for a in state["plan"]:
            if status[a] == TaskStatus.PENDING and any(
                    status.get(d) == TaskStatus.FAILED for d in DEPENDENCIES[a] if d in state["plan"]):
                status[a] = TaskStatus.SKIPPED
                logs.append(LogEntry(agent=a, event="skipped", detail="upstream agent failed"))
        return {"status": status, "logs": logs}

    def validate_node(state: WorkflowState):
        missing = []
        for a in state["plan"]:
            if state["status"][a] != TaskStatus.DONE:
                missing.append(f"{a}: {state['status'][a].value} {state['errors'].get(a, '')}".strip())
        r = state["results"]
        if "competitor" in r and not r["competitor"].competitors:
            missing.append("competitor: no competitors identified")
        if "influencer" in r and not r["influencer"].influencers:
            missing.append("influencer: no verifiable influencers found")
        return {"missing": missing,
                "logs": [LogEntry(agent="orchestrator", event="validate", detail="; ".join(missing) or "ok")]}

    def finalize(state: WorkflowState):
        final = {"brief": state["brief"].model_dump(),
                 "status": {k: v.value for k, v in state["status"].items()},
                 "missing": state["missing"],
                 **{k: v.model_dump(mode="json") for k, v in state["results"].items()}}
        return {"final": final}

    g = StateGraph(WorkflowState)
    g.add_node("plan", plan_node)
    for a in AGENT_ORDER:
        g.add_node(a, make_node(a))
    g.add_node("skip_blocked", skip_blocked)
    g.add_node("router", lambda s: {})
    g.add_node("validate", validate_node)
    g.add_node("finalize", finalize)
    g.set_entry_point("plan")
    g.add_edge("plan", "router")
    targets = {a: a for a in AGENT_ORDER} | {"skip_blocked": "skip_blocked", "validate": "validate"}
    g.add_conditional_edges("router", route, targets)
    for a in AGENT_ORDER:
        g.add_edge(a, "router")
    g.add_edge("skip_blocked", "router")
    g.add_edge("validate", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
