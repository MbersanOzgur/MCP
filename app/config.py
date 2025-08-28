from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables from .env file at project root
load_dotenv()

class Settings(BaseModel):
    # === Gemini LLM Settings ===
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-1.5-flash")

    # === MCP Server Settings ===
    MCP_URL: str = os.getenv("MCP_URL", "http://localhost:8080/")

    # === Server Config ===
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    PORT: int = int(os.getenv("PORT", "8080"))

# Instantiate settings so you can import it elsewhere
settings = Settings()
