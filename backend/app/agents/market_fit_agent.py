from .base import BaseAgent, Evidence, strip_unknown_urls
from ..schemas.agents import MarketFitReport


class MarketFitAgent(BaseAgent):
    name = "product_market_fit"
    schema = MarketFitReport

    def task_prompt(self, brief, context):
        return ("Assess evidence of market opportunity. Provide segments, 3+ personas where evidence permits, "
                "pain points, demand signals, alternatives, risks, hypotheses, validation roadmap and a "
                "go/no-go with rationale. Do NOT claim product-market fit is proven. Separate verified "
                "evidence, inferred signals, assumptions and hypotheses via each finding's level. "
                "State what interviews, surveys or usage data are still needed.")

    def post_validate(self, report: MarketFitReport, ev: Evidence):
        for f in report.pain_points + report.demand_signals:
            f.source_urls = strip_unknown_urls(f.source_urls, ev.urls)
            if f.level.value == "verified" and not f.source_urls:
                f.level = type(f.level).INFERRED  # cannot be verified without a source
        for p in report.personas:
            p.source_urls = strip_unknown_urls(p.source_urls, ev.urls)
        return report
