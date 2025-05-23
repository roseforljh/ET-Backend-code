import os
from dotenv import load_dotenv

load_dotenv()

# Application Info
APP_VERSION = "1.9.9.75-websearch-restored"

# Logging
LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "DEBUG").upper()

# API Endpoints and Paths
DEFAULT_OPENAI_API_BASE_URL = os.getenv("DEFAULT_OPENAI_API_BASE_URL", "https://api.openai.com")
GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com"
OPENAI_COMPATIBLE_PATH = "/v1/chat/completions"

# API Keys and IDs
GOOGLE_API_KEY_ENV = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Timeouts and Limits
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "300"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "60.0"))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "200"))
MAX_SSE_LINE_LENGTH = int(os.getenv("MAX_SSE_LINE_LENGTH", f"{1024 * 1024}")) # 1MB

# Web Search Configuration
SEARCH_RESULT_COUNT = int(os.getenv("SEARCH_RESULT_COUNT", "5"))
SEARCH_SNIPPET_MAX_LENGTH = int(os.getenv("SEARCH_SNIPPET_MAX_LENGTH", "200"))

# AI Behavior
THINKING_PROCESS_SEPARATOR = os.getenv("THINKING_PROCESS_SEPARATOR", "--- FINAL ANSWER ---")

# HTTP
COMMON_HEADERS = {"X-Accel-Buffering": "no"}

# Streaming Heuristics
MIN_FLUSH_LENGTH_HEURISTIC = 80 # Characters for text buffering heuristic