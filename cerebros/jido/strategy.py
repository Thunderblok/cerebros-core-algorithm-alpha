"""Strategy and AgentLoop: Signal-driven agent orchestration.

The Strategy module integrates Runic DAG workflows into Jido's agent loop,
enabling agents to process signals through workflows and produce facts.
This is the core integration layer implementing the agent's decision-making
process.

Analogous to JidoRunic's Strategy module which implements Jido.Agent.Strategy.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from cerebros.jido.signal import Signal, SignalType
from cerebros.jido.fact import Fact, SignalFact
from cerebros.runic.workflow import Workflow, WorkflowResult, WorkflowStatus


@dataclass
class SignalMatcher:
    """Conditional execution gate based on signal type patterns.

    Gates workflow execution based on signal type matching, analogous
    to JidoRunic's SignalMatch module.
    """
    patterns: Set[SignalType] = field(default_factory=set)
    custom_predicate: Optional[Callable[[Signal], bool]] = None

    def matches(self, signal: Signal) -> bool:
        if self.custom_predicate:
            return self.custom_predicate(signal)
        if not self.patterns:
            return True
        return signal.type in self.patterns


@dataclass
class Strategy:
    """Embeds a Runic DAG workflow into the Jido agent loop.

    A Strategy maps signal patterns to workflows, enabling the agent
    to respond to different signal types with appropriate workflow
    execution. This is the bridge between reactive signal processing
    and planned DAG-based computation.

    Attributes:
        name: Strategy identifier.
        workflow: The Runic workflow DAG to execute.
        signal_matcher: Pattern matcher for incoming signals.
        context_builder: Function to build workflow context from a signal.
        fact_extractor: Function to extract facts from workflow results.
    """
    name: str = "default_strategy"
    workflow: Optional[Workflow] = None
    signal_matcher: SignalMatcher = field(default_factory=SignalMatcher)
    context_builder: Optional[Callable[[Signal, Dict], Dict]] = None
    fact_extractor: Optional[Callable[[WorkflowResult], List[Fact]]] = None

    def can_handle(self, signal: Signal) -> bool:
        """Check if this strategy can handle the given signal."""
        return self.signal_matcher.matches(signal)

    def process(
        self,
        signal: Signal,
        agent_state: Optional[Dict[str, Any]] = None,
    ) -> List[Fact]:
        """Process a signal through the workflow, producing facts.

        Args:
            signal: The incoming signal to process.
            agent_state: Current agent state to include in context.

        Returns:
            List of Facts produced by workflow execution.
        """
        if self.workflow is None:
            return []

        if self.context_builder:
            context = self.context_builder(signal, agent_state or {})
        else:
            context = {
                "signal": signal,
                "signal_data": signal.data,
                "signal_type": signal.type,
                **(agent_state or {}),
            }

        result = self.workflow.run(context)

        if self.fact_extractor:
            return self.fact_extractor(result)

        facts = []
        if result.status == WorkflowStatus.COMPLETED:
            fact = Fact(
                value=result.context,
                source_signal_id=signal.id,
                producer_id=self.name,
                metadata={
                    "workflow": self.workflow.name,
                    "execution_order": result.execution_order,
                    "duration_ms": result.duration_ms,
                },
            )
            facts.append(fact)

        return facts


@dataclass
class AgentLoop:
    """Signal-driven agent loop that routes signals through strategies.

    The AgentLoop is the top-level orchestrator that receives signals,
    matches them against registered strategies, and dispatches them
    for workflow processing. It maintains agent state and collects
    produced facts for downstream use.

    Analogous to the Jido agent's main processing loop with
    JidoRunic strategy integration.

    Examples:
        >>> agent = AgentLoop(name="nas_agent")
        >>> agent.register_strategy(search_strategy)
        >>> agent.register_strategy(train_strategy)
        >>> facts = agent.process_signal(Signal(type=SignalType.COMMAND, data={...}))
    """
    name: str = "agent"
    strategies: List[Strategy] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    signal_log: List[Signal] = field(default_factory=list)
    fact_store: List[Fact] = field(default_factory=list)
    _handlers: Dict[SignalType, List[Callable]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def register_strategy(self, strategy: Strategy) -> "AgentLoop":
        """Register a strategy for signal processing."""
        self.strategies.append(strategy)
        return self

    def on(self, signal_type: SignalType, handler: Callable) -> "AgentLoop":
        """Register a direct signal handler (bypasses strategy matching)."""
        self._handlers[signal_type].append(handler)
        return self

    def process_signal(self, signal: Signal) -> List[Fact]:
        """Route a signal through matching strategies.

        Args:
            signal: The signal to process.

        Returns:
            List of Facts produced by all matching strategies.
        """
        self.signal_log.append(signal)
        all_facts = []

        for handler in self._handlers.get(signal.type, []):
            handler(signal, self.state)

        for strategy in self.strategies:
            if strategy.can_handle(signal):
                facts = strategy.process(signal, self.state)
                all_facts.extend(facts)
                self.fact_store.extend(facts)

                for fact in facts:
                    if isinstance(fact.value, dict):
                        self.state.update(fact.value)

        return all_facts

    def emit(self, signal: Signal) -> List[Fact]:
        """Convenience method: create and process a signal."""
        return self.process_signal(signal)

    def get_facts_by_producer(self, producer_id: str) -> List[Fact]:
        """Query facts by their producing component."""
        return [f for f in self.fact_store if f.producer_id == producer_id]

    def get_signals_by_type(self, signal_type: SignalType) -> List[Signal]:
        """Query signal log by type."""
        return [s for s in self.signal_log if s.type == signal_type]
