import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from .config.settings import settings
from .database.models import make_session_factory
from .memory.rag import get_embedder
from .tools.llm import OpenAICompatLLM, LLMUnavailable
from .tools.web_search import default_search

logging.basicConfig(level=logging.INFO)


def create_app(sessions=None, llm=None, search=None, embedder=None) -> FastAPI:
    app = FastAPI(title="MarketMind AI")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","),
                       allow_methods=["*"], allow_headers=["*"])
    app.state.sessions = sessions or make_session_factory()
    try:
        app.state.llm = llm or OpenAICompatLLM()
    except LLMUnavailable:
        app.state.llm = None
    app.state.search = search or default_search()
    app.state.embedder = embedder or get_embedder()
    app.include_router(router)

    @app.get("/api/health")
    def health():
        return {"llm_configured": app.state.llm is not None, "search_configured": app.state.search is not None,
                "search_providers": [type(p).__name__ for p in getattr(app.state.search, "providers", [])],
                "embedder": app.state.embedder.name}
    return app


app = create_app()
