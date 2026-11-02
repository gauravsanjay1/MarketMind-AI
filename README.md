# MarketMind AI

An autonomous multi-agent market research and marketing intelligence system. Give it a product; a LangGraph
orchestrator plans the work and coordinates four specialist agents that do real web research, validate their
own output against the evidence, and produce structured deliverables (UI, PDF reports, CSV).

| Agent | Produces |
|---|---|
| Competitor intelligence | landscape, comparison matrix, features, pricing, SWOT, gaps, recommendations |
| Product-market fit | opportunity, segments, personas, pain points, demand signals, risks, hypotheses, validation roadmap, go/no-go |
| Marketing strategy | positioning, value props, GTM, 30-day calendar, LinkedIn/Instagram/blog/SEO/ads, partnerships |
| Influencer discovery | ranked creators, public contact routes, outreach drafts (never sent automatically) |

## Quick start (everything free)

```bash
cp .env.example .env        # add a free LLM key (see below)
# Backend
cd backend && pip install -r requirements.txt
pytest -q                   # 16 tests, no network or keys needed
uvicorn app.main:app --reload
# Frontend (new terminal)
cd frontend && npm install && npm run dev      # http://localhost:3000
```
Or: `docker compose up --build`.

**Free LLM** (any OpenAI-compatible endpoint; set `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`):
Gemini (aistudio.google.com key, default), Groq, OpenRouter free models, or Ollama locally (no key).
**Search:** DuckDuckGo works with no key and is always the fallback; add `TAVILY_API_KEY` (free tier) for better results.
**Embeddings:** local hashing embedder by default (free, lexical). Set `EMBEDDING_MODEL=text-embedding-004` with Gemini for semantic retrieval.

## Run the TaskFlow demo for real
```bash
cd backend && python ../scripts/run_taskflow_demo.py
```
Runs the full workflow with your providers and writes PDFs and the influencer CSV to `reports/`. Or open the UI,
**New project -> Load TaskFlow demo -> Run research workflow**.

## How it stays honest
- Agent source URLs must come from retrieved evidence; anything else is stripped and flagged.
- "Verified" findings without a source are downgraded to "inferred"; market-fit output carries an explicit caveat.
- Influencers must have a profile URL present in the research evidence; follower counts must appear in the evidence text;
  emails must appear on the creator's **own** page (not another creator's); everything else is "Not available".
- Pages are fetched only if robots.txt allows; failures are reported as limitations, never papered over.
- Without a configured LLM the API returns 503 instead of fake output. Outreach is draft/approved/rejected only; the system never sends.

## Influencer discovery (web search only, intensive)
Up to 3 rounds: LLM plans niche/geography queries -> platform-scoped searches (YouTube, Instagram, LinkedIn, blogs)
-> parallel page fetch -> hop to creators' own contact pages -> batched extraction -> strict validation -> repeat with
new queries until 30 validated creators or no new sources. Instagram/LinkedIn usually block bots, so their follower counts
come only from search snippets when present; these rows show weaker verification status.
Expect ~15 searches per round (~45 per run); the free Tavily tier is 1,000/month.

## Layout
`backend/app/{agents,orchestrator,tools,memory,services,api,database,schemas}` · `frontend/{app,components,hooks,lib,types}` · `docs/` · `scripts/`
See `docs/architecture.md`.

## Known limitations
- Not yet verified against live Gemini/Groq/Tavily/DuckDuckGo in the build environment (no keys or open network there); all logic is tested with mocks and a real HTTP smoke test. First live run may need prompt/model tuning, especially with small local models.
- Hashing embeddings are lexical, not semantic. Scanned PDFs need OCR (unsupported).
- Docker files are written but were not built in the build environment.
- No auth/multi-user. SQLite schema changes need a manual migration (use Alembic when moving to PostgreSQL + pgvector).
