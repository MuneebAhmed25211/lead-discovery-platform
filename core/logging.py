"""
core/logging.py
───────────────
Structured logging setup using structlog.

In development: colourful, human-readable console output via Rich.
In production: JSON-line output suitable for log aggregators (Datadog,
               Logtail, Render's built-in log drain).

Every log entry automatically includes:
  - timestamp (ISO 8601)
  - log level
  - logger name (module path)
  - event message
  - any extra key=value kwargs passed to the logger call

Usage:
    from core.logging import get_logger
    logger = get_logger(__name__)

    logger.info("company_researched", domain="example.com", pages_found=4)
    logger.warning("groq_retry", attempt=2, error=str(e))
    logger.error("discovery_failed", query=query, provider="serper")
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from core.config import settings


def _add_log_level(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add the log level string to every event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def _drop_color_message_key(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Uvicorn adds a 'color_message' key that duplicates 'message'.
    Strip it so logs stay clean.
    """
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging() -> None:
    """
    Call once at application startup (in main.py lifespan).

    Configures both structlog AND the standard library logging so that
    uvicorn's access logs also go through structlog processors.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_log_level,
        _drop_color_message_key,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_production:
        # Production: JSON lines — one log event per line
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    else:
        # Development: pretty console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )

    # Configure structlog itself
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Wire stdlib logging through structlog so uvicorn logs are formatted too
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.LOG_LEVEL)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "supabase", "gotrue", "postgrest"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog logger bound to the given module name.

    Call at module level:
        logger = get_logger(__name__)
    """
    return structlog.get_logger(name)
