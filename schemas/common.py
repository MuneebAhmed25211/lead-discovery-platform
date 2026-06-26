"""
schemas/common.py
─────────────────
Shared Pydantic v2 base models used across the application.

All API responses are wrapped in a consistent envelope:
  Success:  {"success": true,  "data": {...}}
  Error:    {"success": false, "error": "CODE", "message": "..."}

This lets the frontend always check response.success before reading data,
regardless of which endpoint was called.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ── Base Config ───────────────────────────────────────────────────────────────

class AppBaseModel(BaseModel):
    """
    Base for all application models.

    - populate_by_name: allows both alias and field name to be used
    - from_attributes: enables ORM-mode (dict → model from Supabase rows)
    - use_enum_values: store enum .value not the enum object
    """

    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        use_enum_values=True,
    )


# ── Response Envelope ─────────────────────────────────────────────────────────

DataT = TypeVar("DataT")


class SuccessResponse(AppBaseModel, Generic[DataT]):
    """Standard success response wrapper."""

    success: bool = True
    data: DataT
    meta: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(AppBaseModel):
    """Standard error response wrapper. Matches LeadDiscoveryError.to_dict()."""

    success: bool = False
    error: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class PaginatedResponse(AppBaseModel, Generic[DataT]):
    """Paginated list response."""

    success: bool = True
    data: list[DataT]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        if self.page_size == 0:
            return 0
        return (self.total + self.page_size - 1) // self.page_size


# ── Common Field Types ────────────────────────────────────────────────────────

class TimestampedModel(AppBaseModel):
    """Mixin for database rows that have created_at."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UUIDModel(AppBaseModel):
    """Mixin for database rows that have a UUID primary key."""

    id: UUID


# ── Health ────────────────────────────────────────────────────────────────────

class ServiceHealth(AppBaseModel):
    status: str        # "ok" | "error"
    detail: str = ""


class HealthResponse(AppBaseModel):
    status: str        # "ok" | "degraded" | "error"
    version: str = "1.0.0"
    environment: str
    services: dict[str, ServiceHealth]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
