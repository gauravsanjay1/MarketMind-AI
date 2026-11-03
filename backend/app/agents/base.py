import logging
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from ..schemas.common import ProjectBrief, Source
from ..tools.llm import LLMClient
from ..tools.web_search import SearchProvider, dedupe_sources
from ..tools.website_extractor import fetch_page, find_stats, PageContent

log = logging.getLogger("marketmind")

SYSTEM_RULES = (
    "You are a rigorous market research analyst. Use ONLY the evidence provided. "
    "Never invent companies, prices, URLs, people, follower counts or emails. "
    "If a fact is not in the evidence, write 'Not publicly available'. "
    "Every source_urls entry must be copied from the evidence list. "
    "Label claims as verified (directly in a source), inferred, assumption or hypothesis."
)


class QueryPlan(BaseModel):
    queries: list[str]


class Evidence(BaseModel):
    sources: list[Source]
    pages: dict[str, PageContent] = {}
    limitations: list[str] = []

    def prompt_block(self, max_chars: int = 12000) -> str:
        parts = []
        for s in self.sources:
            page = self.pages.get(s.url)
            body = page.text[:1500] if page and page.ok else s.snippet
            stats = (page.stats if page and page.ok else []) or find_stats(s.snippet)
            extra = f"\nSTATS_ON_PAGE: {'; '.join(stats)}" if stats else ""
            emails = f"\nEMAILS_ON_PAGE: {', '.join(page.emails)}" if page and page.ok and page.emails else ""
            parts.append(f"URL: {s.url}\nTITLE: {s.title}\nSNIPPET: {s.snippet[:300]}\nCONTENT: {body}{extra}{emails}")
        return "\n---\n".join(parts)[:max_chars]

    @property
    def urls(self) -> set[str]:
        return {s.url for s in self.sources}


class BaseAgent:
    name = "base"
    schema: type[BaseModel]
    max_pages = 6

    def __init__(self, llm: LLMClient, search: SearchProvider):
        self.llm, self.search = llm, search

    # --- overridable -------------------------------------------------
    def plan_queries(self, brief: ProjectBrief, context: dict) -> list[str]:
        plan = self.llm.structured(
            SYSTEM_RULES,
            f"Write 4 web search queries for the {self.name} task.\n"
            f"Brief: {brief.model_dump_json()}\nContext: {str(context)[:2000]}",
            QueryPlan)
        return plan.queries[:5]

    def task_prompt(self, brief: ProjectBrief, context: dict) -> str:
        raise NotImplementedError

    def post_validate(self, report, evidence: Evidence):
        return report

    # --- pipeline ----------------------------------------------------
    def gather(self, brief: ProjectBrief, context: dict, exclude=frozenset()) -> Evidence:
        sources, limits = [], []
        for q in self.plan_queries(brief, context):
            try:
                sources += self.search.search(q)
            except Exception as e:  # report, never fabricate
                limits.append(f"Search failed for '{q}': {e}")
        sources = [s for s in dedupe_sources(sources) if s.url not in exclude][: self.max_pages * 2]
        pages = {}
        to_fetch = sources[: self.max_pages]
        with ThreadPoolExecutor(max_workers=6) as ex:
            fetched = list(ex.map(fetch_page, [s.url for s in to_fetch]))
        for s, p in zip(to_fetch, fetched):
            pages[s.url] = p
            s.fetched_ok = p.ok
            if not p.ok:
                limits.append(f"Could not open {s.url}: {p.error}")
        return Evidence(sources=sources, pages=pages, limitations=limits)

    def run(self, brief: ProjectBrief, context: dict | None = None):
        context = context or {}
        ev = self.gather(brief, context)
        if not ev.sources:
            raise RuntimeError(f"{self.name}: no research evidence could be collected; "
                               f"{'; '.join(ev.limitations) or 'search returned nothing'}")
        report = self.llm.structured(
            SYSTEM_RULES,
            f"{self.task_prompt(brief, context)}\n\nBRIEF:\n{brief.model_dump_json()}\n\n"
            f"UPSTREAM CONTEXT:\n{str(context)[:10000]}\n\nEVIDENCE:\n{ev.prompt_block()}",
            self.schema)
        report = self.post_validate(report, ev)
        return report, ev


def strip_unknown_urls(urls: list[str], allowed: set[str]) -> list[str]:
    return [u for u in urls if u in allowed]
