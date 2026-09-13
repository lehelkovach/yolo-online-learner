from __future__ import annotations

import random
import uuid
from collections.abc import Callable

IdFactory = Callable[[], str]


def sequential_id_factory(prefix: str = "obj", *, width: int = 6) -> IdFactory:
    """Return ``prefix-000001``, ``prefix-000002``, ... for tests and replay."""
    counter = 0

    def _next() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter:0{width}d}"

    return _next


def seeded_uuid_factory(seed: int, *, prefix: str = "") -> IdFactory:
    """Return UUID4-shaped identifiers drawn from a seeded generator.

    Replaying with the same seed reproduces the same identifier sequence.
    """
    rng = random.Random(seed)

    def _next() -> str:
        value = uuid.UUID(int=rng.getrandbits(128), version=4)
        return f"{prefix}{value}"

    return _next


def random_uuid_factory(prefix: str = "") -> IdFactory:
    """Non-reproducible UUIDs for live sessions that do not need replay."""

    def _next() -> str:
        return f"{prefix}{uuid.uuid4()}"

    return _next
