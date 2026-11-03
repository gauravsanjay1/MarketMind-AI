"""Real end-to-end TaskFlow run using YOUR configured providers (reads .env). Nothing is mocked.

    cd backend && python ../scripts/run_taskflow_demo.py
Writes PDFs + influencer CSV into ../reports/ and prints the execution log.
"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from app.database.models import Project, Run, make_session_factory
from app.memory.rag import get_embedder
from app.schemas.agents import Influencer
from app.schemas.common import ProjectBrief
from app.services.reports import build_sections, stamp
from app.services.workflow import execute_run
from app.tools.influencer_search import export_csv
from app.tools.llm import OpenAICompatLLM, LLMUnavailable
from app.tools.report_generator import build_pdf
from app.tools.web_search import default_search

BRIEF = ProjectBrief(
    company_name="TaskFlow", product_name="TaskFlow", industry="SaaS", geography="India", stage="Pre-launch",
    product_description="AI-powered project management platform that plans sprints, assigns work and flags delivery risks.",
    target_audience="Product and engineering teams at Indian startups and mid-size tech companies",
    brand_tone="Confident, practical, friendly",
    objective="Prepare a complete market research and marketing launch strategy: competitors, product-market fit, "
              "marketing strategy, 30-day calendar, and 30 relevant technology influencers with public contact routes.")

try:
    llm = OpenAICompatLLM()
except LLMUnavailable as e:
    sys.exit(f"{e}. Set LLM_API_KEY in .env (free Gemini/Groq key) or point LLM_BASE_URL at local Ollama.")

sessions = make_session_factory()
with sessions() as db:
    p = Project(brief_json=BRIEF.model_dump_json()); db.add(p); db.commit()
    r = Run(project_id=p.id); db.add(r); db.commit(); rid = r.id
t0 = time.time()
execute_run(sessions, rid, llm, default_search(), 2, get_embedder())
with sessions() as db:
    run = db.get(Run, rid)
    print(f"status={run.status} in {time.time()-t0:.0f}s")
    for l in json.loads(run.logs_json):
        print(f"  {l['agent']:<12} {l['event']:<8} {l['detail']}")
    res = run.result
if res.get("error"):
    sys.exit(res["error"])
out = os.path.join(ROOT, "reports"); os.makedirs(out, exist_ok=True)
for sec in ["competitor", "market_fit", "marketing", "influencer", "all"]:
    try:
        title, sections = build_sections(res, sec)
        print("wrote", build_pdf(os.path.join(out, f"taskflow-{sec}.pdf"), title, sections, stamp()))
    except KeyError:
        print(f"skipped {sec}: section did not complete")
inf = res.get("influencer", {}).get("influencers", [])
if inf:
    print("wrote", export_csv([Influencer(**i) for i in inf], os.path.join(out, "taskflow-influencers.csv")), f"({len(inf)} creators)")
print("Gaps:", res.get("missing") or "none")
