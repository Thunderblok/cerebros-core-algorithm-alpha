"""Introspection: Execution history and provenance tracking.

Provides querying capabilities over execution history, ancestry chains,
and execution statistics. Analogous to JidoRunic's Introspection module
which traces fact lineage and generates execution statistics.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cerebros.jido.signal import Signal
from cerebros.jido.fact import Fact
from cerebros.runic.workflow import WorkflowResult


@dataclass
class ExecutionRecord:
    """Record of a single workflow execution.

    Attributes:
        workflow_name: Name of the executed workflow.
        trigger_signal_id: ID of the signal that triggered execution.
        strategy_name: Name of the strategy that handled the signal.
        result_status: Final status of the workflow execution.
        duration_ms: Execution time in milliseconds.
        facts_produced: Number of facts produced.
        execution_order: Component execution order.
        timestamp: When the execution occurred.
        metadata: Additional execution metadata.
    """
    workflow_name: str
    trigger_signal_id: str = ""
    strategy_name: str = ""
    result_status: str = "unknown"
    duration_ms: float = 0.0
    facts_produced: int = 0
    execution_order: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Introspection:
    """Queries execution history and ancestry chains.

    Provides a unified interface for inspecting the execution history
    of an AgentLoop, including fact lineage, signal correlation,
    and performance statistics.

    Examples:
        >>> intro = Introspection(agent_loop)
        >>> intro.execution_count()
        42
        >>> intro.ancestry_chain(fact_id)
        [Fact(...), Fact(...), Fact(...)]
    """

    def __init__(self, agent_loop: Any):
        self.agent_loop = agent_loop
        self._execution_records: List[ExecutionRecord] = []

    def record_execution(
        self,
        workflow_name: str,
        signal: Signal,
        strategy_name: str,
        result: WorkflowResult,
        facts: List[Fact],
    ):
        """Record a workflow execution for later introspection."""
        self._execution_records.append(ExecutionRecord(
            workflow_name=workflow_name,
            trigger_signal_id=signal.id,
            strategy_name=strategy_name,
            result_status=result.status.value,
            duration_ms=result.duration_ms,
            facts_produced=len(facts),
            execution_order=result.execution_order,
        ))

    def execution_count(self) -> int:
        return len(self._execution_records)

    def total_duration_ms(self) -> float:
        return sum(r.duration_ms for r in self._execution_records)

    def average_duration_ms(self) -> float:
        if not self._execution_records:
            return 0.0
        return self.total_duration_ms() / len(self._execution_records)

    def ancestry_chain(self, fact_id: str) -> List[Fact]:
        """Trace the full lineage of a fact through its ancestry."""
        fact_index = {f.id: f for f in self.agent_loop.fact_store}
        target = fact_index.get(fact_id)
        if not target:
            return []

        chain = []
        for ancestor_id in target.ancestry:
            ancestor = fact_index.get(ancestor_id)
            if ancestor:
                chain.append(ancestor)
        chain.append(target)
        return chain

    def correlated_signals(self, correlation_id: str) -> List[Signal]:
        """Find all signals sharing a correlation ID."""
        return [
            s for s in self.agent_loop.signal_log
            if s.correlation_id == correlation_id
        ]

    def facts_by_signal(self, signal_id: str) -> List[Fact]:
        """Find all facts produced from a given signal."""
        return [
            f for f in self.agent_loop.fact_store
            if f.source_signal_id == signal_id
        ]

    def execution_stats(self) -> Dict[str, Any]:
        """Generate aggregate execution statistics."""
        if not self._execution_records:
            return {"count": 0}

        durations = [r.duration_ms for r in self._execution_records]
        statuses = {}
        for r in self._execution_records:
            statuses[r.result_status] = statuses.get(r.result_status, 0) + 1

        return {
            "count": len(self._execution_records),
            "total_duration_ms": sum(durations),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "total_facts_produced": sum(
                r.facts_produced for r in self._execution_records
            ),
            "status_counts": statuses,
        }

    def get_records(
        self,
        strategy_name: Optional[str] = None,
        workflow_name: Optional[str] = None,
    ) -> List[ExecutionRecord]:
        """Query execution records with optional filters."""
        records = self._execution_records
        if strategy_name:
            records = [r for r in records if r.strategy_name == strategy_name]
        if workflow_name:
            records = [r for r in records if r.workflow_name == workflow_name]
        return records
