"""Signal: Event/message representation for the Jido agent system.

Signals are the primary communication mechanism in Jido's agent architecture.
They represent events, commands, or data flowing through the system.
Signals can trigger workflow execution, gate conditional logic, and
carry provenance information for execution tracing.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalType(Enum):
    """Classification of signals in the agent system."""
    COMMAND = "command"
    EVENT = "event"
    QUERY = "query"
    RESULT = "result"
    ERROR = "error"

    # NAS-specific signal types
    ARCHITECTURE_GENERATED = "architecture_generated"
    MODEL_BUILT = "model_built"
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    EVALUATION_COMPLETED = "evaluation_completed"
    SEARCH_COMPLETED = "search_completed"


@dataclass
class Signal:
    """An event or message flowing through the Jido agent system.

    Signals carry typed data between components and agents, supporting
    causality tracking and pattern-based routing. Analogous to signals
    in the Elixir Jido library.

    Attributes:
        type: Classification of this signal.
        data: Payload data carried by the signal.
        source: Identifier of the component that created this signal.
        correlation_id: Groups related signals together.
        causation_id: ID of the signal that caused this one.
        timestamp: When the signal was created.
        metadata: Additional key-value metadata.
    """
    type: SignalType
    data: Any = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = ""
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def derive(
        self,
        signal_type: SignalType,
        data: Any = None,
        source: str = "",
        **metadata,
    ) -> "Signal":
        """Create a derived signal maintaining causality chain.

        The new signal's causation_id points to this signal,
        and shares the same correlation_id.
        """
        return Signal(
            type=signal_type,
            data=data,
            source=source or self.source,
            correlation_id=self.correlation_id or self.id,
            causation_id=self.id,
            metadata={**self.metadata, **metadata},
        )

    def matches(self, pattern: SignalType) -> bool:
        """Check if this signal matches a type pattern."""
        return self.type == pattern

    def __repr__(self) -> str:
        return (
            f"Signal(type={self.type.value}, source={self.source!r}, "
            f"id={self.id})"
        )
