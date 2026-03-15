"""Cerebros Jido: Signal-driven agent orchestration framework.

A Python port of the core concepts from the Elixir Jido/JidoRunic libraries
(https://github.com/agentjido/jido_runic), adapted for the Cerebros NAS framework.

Jido provides signal-driven agent architecture where:
- Signals represent events/messages flowing through the system
- Facts represent derived knowledge from signal processing
- ActionNodes wrap domain actions as executable workflow nodes
- Strategies define how agents process signals through workflows
- Introspection enables execution history and provenance tracking
"""

from cerebros.jido.signal import Signal, SignalType
from cerebros.jido.fact import Fact, SignalFact
from cerebros.jido.action_node import ActionNode
from cerebros.jido.strategy import Strategy, AgentLoop
from cerebros.jido.introspection import Introspection, ExecutionRecord

__all__ = [
    "Signal",
    "SignalType",
    "Fact",
    "SignalFact",
    "ActionNode",
    "Strategy",
    "AgentLoop",
    "Introspection",
    "ExecutionRecord",
]
