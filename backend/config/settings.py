# config/settings.py
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog
import logging
import sys

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Settings
    API_VERSION: str = "v1"
    API_PREFIX: str = f"/api/{API_VERSION}"
    DEBUG: bool = False
    AUTH_SECRET: Optional[str] = None
    
    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    # Database Settings
    SQLITE_DB_URL: str = "sqlite:///checkpoints.db"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    JSON_LOGS: bool = True
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True
    )
    
    def get_available_models(self) -> list[str]:
        """Get list of available models based on API keys"""
        models = []
        if self.OPENAI_API_KEY:
            models.extend(["gpt-4", "gpt-3.5-turbo"])
        if self.ANTHROPIC_API_KEY:
            models.extend(["claude-3-opus", "claude-3-sonnet"])
        if self.GOOGLE_API_KEY:
            models.append("gemini-pro")
        if self.GROQ_API_KEY:
            models.append("llama-70b")
        return models

# logger.py
def setup_logging(settings: Settings) -> None:
    """Configure structured logging"""
    
    # Set logging level
    log_level = getattr(logging, settings.LOG_LEVEL.upper())
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer() if settings.JSON_LOGS else structlog.dev.ConsoleRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Add formatter if using JSON logs
    if settings.JSON_LOGS:
        handler.setFormatter(logging.Formatter('%(message)s'))
    
    root_logger.addHandler(handler)

# Get logger
logger = structlog.get_logger()

# metrics.py
from prometheus_client import Counter, Histogram

# Define metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

model_requests_total = Counter(
    'model_requests_total',
    'Total number of model requests',
    ['model_name', 'status']
)

model_tokens_total = Counter(
    'model_tokens_total',
    'Total number of tokens processed',
    ['model_name', 'type']  # type can be 'input' or 'output'
)

model_request_duration_seconds = Histogram(
    'model_request_duration_seconds',
    'Model request duration in seconds',
    ['model_name']
)