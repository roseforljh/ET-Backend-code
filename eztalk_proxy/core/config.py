import os
from dotenv import load_dotenv

load_dotenv()

APP_VERSION = os.getenv("APP_VERSION", "1.9.9.77-gcs-support")

LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "INFO").upper()

DEFAULT_OPENAI_API_BASE_URL = os.getenv("DEFAULT_OPENAI_API_BASE_URL", "https://api.openai.com")
GOOGLE_API_BASE_URL = os.getenv("GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com")
OPENAI_COMPATIBLE_PATH = os.getenv("OPENAI_COMPATIBLE_PATH", "/v1/chat/completions")

GOOGLE_API_KEY_ENV = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "600"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "60.0"))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "200"))
MAX_SSE_LINE_LENGTH = int(os.getenv("MAX_SSE_LINE_LENGTH", f"{1024 * 1024}"))

SEARCH_RESULT_COUNT = int(os.getenv("SEARCH_RESULT_COUNT", "5"))
SEARCH_SNIPPET_MAX_LENGTH = int(os.getenv("SEARCH_SNIPPET_MAX_LENGTH", "200"))

THINKING_PROCESS_SEPARATOR = os.getenv("THINKING_PROCESS_SEPARATOR", "--- FINAL ANSWER ---")
MIN_FLUSH_LENGTH_HEURISTIC = int(os.getenv("MIN_FLUSH_LENGTH_HEURISTIC", "80"))

COMMON_HEADERS = {"X-Accel-Buffering": "no"}

TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "/tmp/temp_document_uploads")
MAX_DOCUMENT_UPLOAD_SIZE_MB = int(os.getenv("MAX_DOCUMENT_UPLOAD_SIZE_MB", "20"))
MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT = int(os.getenv("MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT", "15000"))

SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION = [
    # Plain Text & Data Formats
    "text/plain",
    "text/html",
    "text/csv",
    "text/markdown",
    "text/x-markdown",
    "application/json",
    "text/xml",
    "application/xml",
    "text/rtf",
    "application/rtf",
    
    # Microsoft Office Documents
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
    "application/msword", # .doc
    
    # Excel Documents
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", # .xlsx
    "application/vnd.ms-excel", # .xls
    "text/csv", # .csv (already included above)
    
    # PowerPoint Documents  
    "application/vnd.openxmlformats-officedocument.presentationml.presentation", # .pptx
    "application/vnd.ms-powerpoint", # .ppt
    
    # Open Office Documents
    "application/vnd.oasis.opendocument.text", # .odt
    "application/vnd.oasis.opendocument.spreadsheet", # .ods
    "application/vnd.oasis.opendocument.presentation", # .odp
    
    # E-book Formats
    "application/epub+zip", # .epub
    
    # Archive Formats (text extraction from contained files)
    # "application/zip", # .zip (uncomment if needed)
    # "application/x-rar-compressed", # .rar (uncomment if needed)

    # Audio Formats (for transcription)
    "audio/flac",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg", # .mp3
    "audio/mp4", # .m4a
    "audio/aac",
    "audio/ogg",
    "audio/webm",

    # Common Code & Configuration Formats
    "application/x-javascript",
    "text/javascript",
    "text/css",
    "application/x-python",
    "text/x-python",
    "application/x-yaml",
    "text/yaml",
    "text/x-yaml",
    "application/toml",
    "text/x-toml",
    "application/x-sh",
    "text/x-shellscript",
    "application/x-php",
    "text/x-php",
    "application/x-ruby",
    "text/x-ruby",
    "application/x-perl",
    "text/x-perl",
    "application/x-httpd-php",
    "text/x-c",
    "text/x-c++",
    "text/x-java-source",
    "text/x-csharp",
    
    # Log & Config Files
    "text/x-log",
    "application/x-log",
    "text/x-ini",
    "application/x-wine-extension-ini",
]

GEMINI_SUPPORTED_UPLOAD_MIMETYPES = [
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    "video/mp4", "application/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-flv",
    "video/x-matroska",
    "video/webm",
    "video/x-ms-wmv",
    "video/3gpp",
    "video/x-m4v",
    "audio/wav", "audio/x-wav",
    "audio/mpeg",
    "audio/aac",
    "audio/ogg",
    "audio/opus",
    "audio/flac",
    "audio/midi",
    "audio/amr",
    "audio/aiff",
    "audio/x-m4a",
    "text/plain",
    "application/pdf",
]

GEMINI_ENABLE_GCS_UPLOAD = os.getenv("GEMINI_ENABLE_GCS_UPLOAD", "False").lower() == "true"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", None)

# Cloudflare bypass strategy for third-party Gemini proxies
# Options: "full" (all browser headers), "minimal" (basic headers only), "none" (no extra headers)
CLOUDFLARE_BYPASS_STRATEGY = os.getenv("CLOUDFLARE_BYPASS_STRATEGY", "full").lower()

# ===== Image Generation Presets (Default/SiliconFlow) =====
# 前端"默认"平台：隐藏参数，后端自动注入；密钥仅从本地环境读取，禁止提交到仓库
SILICONFLOW_IMAGE_API_URL = os.getenv(
    "SILICONFLOW_IMAGE_API_URL",
    "https://api.siliconflow.cn/v1/images/generations"
)
SILICONFLOW_DEFAULT_IMAGE_MODEL = os.getenv(
    "SILICONFLOW_DEFAULT_IMAGE_MODEL",
    "Kwai-Kolors/Kolors"
)
# 支持两种变量名，便于本地配置：SILICONFLOW_API_KEY 或 SILICONFLOW_DEFAULT_API_KEY
SILICONFLOW_API_KEY_DEFAULT = os.getenv("SILICONFLOW_API_KEY") or os.getenv("SILICONFLOW_DEFAULT_API_KEY")

# ===== Text Chat Presets (Default) =====
# 前端"默认"平台：隐藏参数，后端自动注入；密钥仅从本地环境读取，禁止提交到仓库
DEFAULT_TEXT_API_URL = os.getenv(
    "DEFAULT_TEXT_API_URL",
    "https://ai.luuu71.dpdns.org/v1/chat/completions"
)
DEFAULT_TEXT_API_KEY = os.getenv(
    "DEFAULT_TEXT_API_KEY",
    ""  # 从环境变量读取，不要硬编码
)
DEFAULT_TEXT_MODELS = os.getenv(
    "DEFAULT_TEXT_MODELS",
    "gemini-2.5-pro-1M,gemini-2.5-flash,gemini-flash-lite-latest"
).split(",")