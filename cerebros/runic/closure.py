"""Closure and ClosureMetadata: Content-addressable function wrappers.

Inspired by Runic's closure system which provides deterministic hashing
of functions and their bound variables for reproducible workflow execution.
"""

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class ClosureMetadata:
    """Metadata for a closure, enabling content-addressable identification.

    Attributes:
        name: Human-readable name for this closure.
        source_hash: SHA256 hash of the function source code.
        bindings: Frozen dict of captured variable bindings.
        arity: Number of arguments the function accepts.
    """
    name: str
    source_hash: str
    bindings: Tuple[Tuple[str, Any], ...] = ()
    arity: int = 0

    @property
    def content_hash(self) -> str:
        raw = f"{self.source_hash}:{self.bindings}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Closure:
    """A content-addressable function wrapper with metadata.

    Wraps a callable with metadata that supports deterministic hashing,
    serialization, and reproducible execution. Analogous to Runic's
    closure system with pin-operator variable capture.

    Args:
        func: The callable to wrap.
        name: Optional human-readable name (defaults to func.__name__).
        bindings: Dict of captured outer-scope variable bindings.
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        bindings: Optional[Dict[str, Any]] = None,
    ):
        self.func = func
        self.bindings = bindings or {}
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            source = repr(func)
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
        sig = inspect.signature(func)
        arity = len(sig.parameters)

        self.metadata = ClosureMetadata(
            name=name or getattr(func, "__name__", "anonymous"),
            source_hash=source_hash,
            bindings=tuple(sorted(self.bindings.items())),
            arity=arity,
        )

    def __call__(self, *args, **kwargs):
        merged = {**self.bindings, **kwargs}
        return self.func(*args, **merged)

    @property
    def content_hash(self) -> str:
        return self.metadata.content_hash

    @property
    def name(self) -> str:
        return self.metadata.name

    def __repr__(self) -> str:
        return f"Closure({self.name}, hash={self.content_hash})"
