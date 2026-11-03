from .base import BaseAgent
from ..schemas.agents import MarketingReport


class MarketingAgent(BaseAgent):
    name = "marketing_strategy"
    schema = MarketingReport
    max_pages = 3

    def task_prompt(self, brief, context):
        return ("Build a marketing strategy grounded in the upstream competitor and customer research: "
                "positioning, value propositions, audience strategy, channels, go-to-market, a 30-day content "
                "calendar (days 1-30), LinkedIn drafts, Instagram ideas, blog topics, SEO keywords, ad concepts "
                "and partnerships. Reference specific competitor gaps and personas. Avoid generic content.")
