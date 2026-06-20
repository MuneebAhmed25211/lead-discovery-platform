"""
cache/base.py
─────────────
Abstract cache interface.

Defines the contract that all cache implementations must satisfy.
Currently only SupabaseCache is implemented, but this abstraction
makes it trivial to swap to Redis/Upstash when traffic grows.

All keys are namespaced strings.
All values are JSON-serialisable dicts.
All TTLs are in hours.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """
        Return the cached value for key, or None if not found / expired.
        Must never raise — return None on any error.
        """
        ...

    @abstractmethod
    async def set(self, key: str, value: dict[str, Any], ttl_hours: int) -> bool:
        """
        Store value under key with a TTL in hours.
        Returns True on success, False on failure.
        Must never raise.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if deleted, False otherwise."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if key exists and is not expired."""
        ...
