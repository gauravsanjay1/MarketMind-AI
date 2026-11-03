from .base import BaseAgent, Evidence, strip_unknown_urls
from ..schemas.agents import CompetitorReport


class CompetitorAgent(BaseAgent):
    name = "competitor_intelligence"
    schema = CompetitorReport

    def task_prompt(self, brief, context):
        return ("Identify direct, indirect and emerging competitors. For each: features, public pricing, "
                "positioning, audience, strengths, weaknesses. Then SWOT (keys: strengths, weaknesses, "
                "opportunities, threats), competitive gaps, differentiation and recommendations.")

    def post_validate(self, report: CompetitorReport, ev: Evidence):
        for c in report.competitors:
            c.source_urls = strip_unknown_urls(c.source_urls, ev.urls)
            if not c.source_urls:  # unsupported -> flag, do not silently trust
                c.pricing = "Not publicly available"
                report.limitations.append(f"{c.name}: no retrieved source supports this profile")
        for f in report.gaps + report.differentiation:
            f.source_urls = strip_unknown_urls(f.source_urls, ev.urls)
        report.limitations += ev.limitations
        return report
