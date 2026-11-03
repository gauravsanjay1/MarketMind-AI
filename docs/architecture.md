# Architecture

```
Next.js UI --REST--> FastAPI --BackgroundTask--> LangGraph workflow
                                   |                 plan -> router -> {competitor, market_fit} -> marketing -> influencer -> validate -> finalize
                                   |                                      (deps enforced; retries; failed upstream => dependents skipped)
                                   +-- SQLite: projects, runs (temporary workflow state), documents/chunks (RAG), memory_items (long-term)
Agents -> tools: SearchProvider (Tavily -> DuckDuckGo chain) | website_extractor (robots.txt, BeautifulSoup) | LLMClient (OpenAI-compatible) | reports (ReportLab) | CSV (pandas)
```

**State & routing** (`orchestrator/`): typed `WorkflowState`; `plan` selects agents and adds transitive dependencies;
`router` picks the next runnable agent or `skip_blocked`; each agent node retries (`max_retries`) then marks FAILED;
`validate` lists gaps; `finalize` assembles the result. Progress is streamed to the DB after every step; the UI polls.

**Agent pipeline** (`agents/base.py`): plan queries -> search -> dedupe -> parallel fetch -> structured LLM synthesis
(Pydantic schema, JSON-mode, one self-repair retry) -> `post_validate` (evidence enforcement).

**Memory vs workflow state:** `Run` rows hold temporary results; `MemoryItem` + `Document/Chunk` hold long-term company
knowledge (brand guidelines, decisions, previous findings, uploaded docs). Each run injects memory plus top-5 retrieved
chunks (cited as `[doc:file#chunk]`) into agent context.

**PostgreSQL + pgvector path:** set `DATABASE_URL=postgresql+psycopg://...`, change `Chunk.embedding_json` to a
`pgvector.sqlalchemy.Vector(dim)` column, replace the numpy scan in `memory/rag.py:retrieve` with
`ORDER BY embedding <=> :q LIMIT k`, and adopt Alembic migrations.

**Swapping providers:** implement `SearchProvider.search()`, `LLMClient.structured()` or `Embedder.embed()`.
