"""
cache/supabase_cache.py
───────────────────────
Supabase Postgres-backed cache implementation.

Uses a dedicated `cache` table with columns:
  key text PRIMARY KEY
  value jsonb NOT NULL
  expires_at timestamptz NOT NULL
  created_at timestamptz DEFAULT now()

Why Supabase instead of Redis?
  - At portfolio traffic volume, Redis is overkill and costs extra money.
  - Supabase is already in the stack — no additional service to manage.
  - A simple SELECT + expiry check is fast enough for our access patterns.
  - When traffic grows, we swap to Upstash Redis by replacing this class.

All methods catch every exception and log a warning rather than raising.
Cache errors are NON-FATAL — the application continues without cached data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from supabase import Client

from cache.base import BaseCache
from core.errors import CacheError
from core.logging import get_logger

logger = get_logger(__name__)

TABLE = "cache"


class SupabaseCache(BaseCache):

    def __init__(self, client: Client) -> None:
        self._db = client

    # ── Public interface ──────────────────────────────────────────────────────

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            result = (
                self._db.table(TABLE)
                .select("value, expires_at")
                .eq("key", key)
                .single()
                .execute()
            )

            if not result.data:
                return None

            row = result.data
            expires_at = datetime.fromisoformat(row["expires_at"])

            # Normalise to UTC-aware for comparison
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

            if datetime.now(timezone.utc) > expires_at:
                # Expired — clean it up asynchronously (best-effort)
                self._delete_sync(key)
                return None

            logger.debug("cache_hit", key=key)
            return row["value"]

        except Exception as exc:
            # PostgREST returns an error if no row found — treat as miss
            error_msg = str(exc)
            if "no rows" in error_msg.lower() or "PGRST116" in error_msg:
                return None
            logger.warning("cache_get_error", key=key, error=error_msg)
            return None

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_hours: int,
    ) -> bool:
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)

            self._db.table(TABLE).upsert(
                {
                    "key": key,
                    "value": value,
                    "expires_at": expires_at.isoformat(),
                },
                on_conflict="key",
            ).execute()

            logger.debug("cache_set", key=key, ttl_hours=ttl_hours)
            return True

        except Exception as exc:
            logger.warning("cache_set_error", key=key, error=str(exc))
            return False

    async def delete(self, key: str) -> bool:
        return self._delete_sync(key)

    async def exists(self, key: str) -> bool:
        result = await self.get(key)
        return result is not None

    # ── Key builders ──────────────────────────────────────────────────────────

    @staticmethod
    def make_key(namespace: str, *parts: str) -> str:
        """
        Build a deterministic cache key.

        For short, stable values: join directly.
        For variable-length inputs (descriptions, queries): SHA-256 hash.

        Examples:
          make_key("icp", description)
            → "icp:{sha256(description)}"
          make_key("p1", "example.com", "/about")
            → "p1:example.com:/about"
          make_key("score", company_id)
            → "score:{company_id}"
        """
        if len(parts) == 1 and len(parts[0]) > 64:
            # Long string → hash it so the key stays under 200 chars
            hashed = hashlib.sha256(parts[0].encode()).hexdigest()[:16]
            return f"{namespace}:{hashed}"
        return f"{namespace}:{':'.join(parts)}"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _delete_sync(self, key: str) -> bool:
        try:
            self._db.table(TABLE).delete().eq("key", key).execute()
            return True
        except Exception as exc:
            logger.warning("cache_delete_error", key=key, error=str(exc))
            return False


def get_cache(client: Client) -> SupabaseCache:
    """Factory — returns a SupabaseCache bound to the given client."""
    return SupabaseCache(client)
