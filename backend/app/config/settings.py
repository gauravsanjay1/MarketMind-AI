from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # Any OpenAI-compatible endpoint works (Gemini, Groq, OpenRouter, Ollama, OpenAI)
    llm_api_key: str = ""
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-2.0-flash"
    tavily_api_key: str = ""
    embedding_model: str = ""   # e.g. text-embedding-004 (Gemini); empty = local hashing embedder
    cors_origins: str = "http://localhost:3000"
    database_url: str = "sqlite:///./data/marketmind.db"
    max_retries: int = 2


settings = Settings()
