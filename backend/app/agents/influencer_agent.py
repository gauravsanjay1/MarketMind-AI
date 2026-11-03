"""Intensive, multi-round, web-search-only influencer discovery.

Rounds: LLM plans niche/geo queries -> platform-scoped searches -> parallel page fetch ->
contact-page hop on creators' own sites -> batched extraction -> strict evidence validation.
Repeats with new queries until the target count of validated creators is reached.
"""
from urllib.parse import urlparse
from .base import BaseAgent, Evidence, QueryPlan, SYSTEM_RULES
from ..schemas.agents import InfluencerReport, Influencer
from ..schemas.common import ProjectBrief, Source
from ..tools.influencer_search import classify_tier, dedupe_influencers
from ..tools.web_search import normalize_url
from ..tools.website_extractor import fetch_page, is_social

PLATFORMS = ["site:youtube.com", "site:instagram.com", "site:linkedin.com/in"]


def _followers_in_text(n: int, text: str) -> bool:
    forms = {f"{n:,}", str(n)}
    if n >= 1000:
        forms.add(f"{n/1000:g}k")
    if n >= 1_000_000:
        forms.add(f"{n/1_000_000:g}m")
    low = text.lower()
    return any(f.lower() in low for f in forms)


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _emails_for(i: Influencer, ev: Evidence) -> set[str]:
    """Emails that appear on the creator's OWN page(s): exact profile page, or same-host page on a
    non-shared (personal) domain. Never emails from other creators' pages."""
    out = set()
    for url, p in ev.pages.items():
        if not p.ok:
            continue
        own = normalize_url(url) == normalize_url(i.profile_url) or \
            (not is_social(url) and _host(url) == _host(i.profile_url))
        if own:
            out |= {e.lower() for e in p.emails}
    return out


class InfluencerAgent(BaseAgent):
    name = "influencer_discovery"
    schema = InfluencerReport
    max_pages = 20
    max_rounds = 3
    target = 30
    batch_size = 6

    def plan_queries(self, brief: ProjectBrief, context: dict) -> list[str]:
        plan = self.llm.structured(
            SYSTEM_RULES,
            "Write 5 web search queries to discover creators/influencers who could promote this product to its "
            "target audience (use the positioning/personas in context). Vary niches, sub-niches and geographies. "
            f"Round {context.get('round', 1)}. Already found (find DIFFERENT ones): {context.get('already_found', [])}\n"
            f"Brief: {brief.model_dump_json()}\nContext: {str(context)[:2500]}", QueryPlan)
        qs = plan.queries[:5]
        extra = [f"{q} {p}" for q in qs[:3] for p in PLATFORMS]
        extra.append(f"{qs[0]} creator collaboration business email contact" if qs else "")
        return qs + [e for e in extra if e]

    def task_prompt(self, brief, context):
        return ("From the evidence ONLY, list creators relevant to the target audience and positioning. "
                "profile_url and source_url must come from the evidence. Leave followers, engagement, location, "
                "public_email and contact_url null unless explicitly present in the evidence (see STATS_ON_PAGE / "
                "EMAILS_ON_PAGE). Give priority_score 0-100 based on niche match, geography and audience fit, and a "
                "collaboration_type. Draft one personalised email and one DM per creator referencing their actual "
                "content (approval_status 'draft'). Never claim a message was sent.")

    def _follow_contacts(self, ev: Evidence):
        """One-hop to the creator's own contact/collab page (same host, robots.txt respected)."""
        for url, page in list(ev.pages.items()):
            if not page.ok or is_social(url):
                continue
            for link in page.contact_links:
                if _host(link) == _host(url) and link not in ev.pages:
                    c = fetch_page(link)
                    ev.pages[link] = c
                    ev.sources.append(Source(url=link, title="Contact page", snippet="", fetched_ok=c.ok))
                    if not c.ok:
                        ev.limitations.append(f"Could not open {link}: {c.error}")
                    break

    def run(self, brief: ProjectBrief, context: dict | None = None):
        context = dict(context or {})
        seen: dict[str, Source] = {}
        pages: dict = {}
        limits: list[str] = []
        notes: list[str] = []
        items: list[Influencer] = []
        drafts = []
        report, full = None, None
        for rnd in range(1, self.max_rounds + 1):
            ev = self.gather(brief, {**context, "round": rnd, "already_found": [i.name for i in items]},
                             exclude=frozenset(seen))
            self._follow_contacts(ev)
            limits += ev.limitations
            fresh = [s for s in ev.sources if s.url not in seen]
            if not fresh:
                break
            for s in fresh:
                seen[s.url] = s
            pages.update(ev.pages)
            for k in range(0, len(fresh), self.batch_size):
                batch = fresh[k:k + self.batch_size]
                sub = Evidence(sources=batch, pages={s.url: pages[s.url] for s in batch if s.url in pages})
                try:
                    part = self.llm.structured(
                        SYSTEM_RULES,
                        f"{self.task_prompt(brief, context)}\n\nBRIEF:\n{brief.model_dump_json()}\n\n"
                        f"UPSTREAM CONTEXT:\n{str(context)[:6000]}\n\nEVIDENCE:\n{sub.prompt_block()}",
                        self.schema)
                except Exception as e:
                    limits.append(f"Extraction batch failed ({type(e).__name__}): {e}")
                    continue
                items += part.influencers
                drafts += part.outreach
            full = Evidence(sources=list(seen.values()), pages=pages, limitations=list(dict.fromkeys(limits)))
            report = self.post_validate(InfluencerReport(influencers=items, outreach=drafts), full)
            notes += [n for n in report.limitations if n not in notes]
            items, drafts = list(report.influencers), list(report.outreach)
            if len(items) >= self.target:
                break
        if report is None:
            raise RuntimeError(f"{self.name}: no research evidence could be collected; "
                               f"{'; '.join(dict.fromkeys(limits)) or 'search returned nothing'}")
        if len(report.influencers) < self.target:
            notes.append(f"Found {len(report.influencers)} verifiable creators; target was {self.target}. "
                         "Remaining candidates could not be supported by public evidence.")
        report.limitations = notes
        return report, full

    def post_validate(self, report: InfluencerReport, ev: Evidence):
        corpus = " ".join(p.text for p in ev.pages.values() if p.ok) + " " + \
                 " ".join(" ".join(p.stats) for p in ev.pages.values() if p.ok) + " " + \
                 " ".join(s.snippet for s in ev.sources)
        kept, notes = [], []
        for i in report.influencers:
            if i.profile_url not in ev.urls and i.profile_url not in corpus:
                notes.append(f"Dropped {i.name}: profile URL not found in research evidence")
                continue
            if i.public_email and i.public_email.lower() not in _emails_for(i, ev):
                i.public_email = None
            if i.contact_url and i.contact_url not in ev.urls:
                i.contact_url = None
            if i.followers and not _followers_in_text(i.followers, corpus):
                i.followers = None
            i.tier = classify_tier(i.followers)
            opened = i.profile_url in ev.pages and ev.pages[i.profile_url].ok
            i.verification_status = ("profile_page_opened" if opened else
                                     "profile_url_found_in_search" if i.profile_url in ev.urls else
                                     "mentioned_in_sources")
            kept.append(i)
        report.influencers = sorted(dedupe_influencers(kept), key=lambda x: -x.priority_score)
        names = {i.name for i in report.influencers}
        report.outreach = [o for o in report.outreach if o.influencer_name in names]
        for o in report.outreach:
            o.approval_status = "draft"
        report.limitations = notes
        return report
