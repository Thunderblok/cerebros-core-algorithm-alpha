"""Fact and SignalFact: Derived knowledge from signal processing.

Facts represent derived knowledge produced by processing signals through
workflows. The SignalFact adapter provides bidirectional translation between
signals and facts while maintaining causality information.

Analogous to JidoRunic's SignalFact module.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cerebros.jido.signal import Signal, SignalType


@dataclass
class Fact:
    """A piece of derived knowledge from workflow execution.

    Facts are the outputs of workflow processing - they represent
    conclusions, computed values, or derived state. Facts maintain
    provenance information linking them back to their source signals.

    Attributes:
        value: The derived data/knowledge.
        source_signal_id: ID of the signal that produced this fact.
        producer_id: ID of the component that created this fact.
        confidence: Optional confidence score (0.0 to 1.0).
        ancestry: Chain of fact IDs leading to this one.
        timestamp: When the fact was created.
        metadata: Additional key-value metadata.
    """
    value: Any
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_signal_id: Optional[str] = None
    producer_id: str = ""
    confidence: float = 1.0
    ancestry: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def derive(self, value: Any, producer_id: str = "", **metadata) -> "Fact":
        """Create a derived fact maintaining lineage."""
        return Fact(
            value=value,
            source_signal_id=self.source_signal_id,
            producer_id=producer_id or self.producer_id,
            ancestry=[*self.ancestry, self.id],
            metadata={**self.metadata, **metadata},
        )


class SignalFact:
    """Bidirectional adapter between Signals and Facts.

    Provides conversion methods maintaining causality information,
    analogous to JidoRunic's SignalFact module.
    """

    @staticmethod
    def signal_to_fact(
        signal: Signal,
        value: Any = None,
        producer_id: str = "",
    ) -> Fact:
        """Convert a Signal into a Fact, preserving provenance."""
        return Fact(
            value=value if value is not None else signal.data,
            source_signal_id=signal.id,
            producer_id=producer_id or signal.source,
            metadata={
                "signal_type": signal.type.value,
                "correlation_id": signal.correlation_id,
                **signal.metadata,
            },
        )

    @staticmethod
    def fact_to_signal(
        fact: Fact,
        signal_type: SignalType = SignalType.RESULT,
        source: str = "",
    ) -> Signal:
        """Convert a Fact back into a Signal for downstream processing."""
        return Signal(
            type=signal_type,
            data=fact.value,
            source=source or fact.producer_id,
            metadata={
                "fact_id": fact.id,
                "ancestry": fact.ancestry,
                "confidence": fact.confidence,
                **fact.metadata,
            },
        )
