"""Cerebros Runic: DAG-based workflow composition engine.

A Python port of the core concepts from the Elixir Runic library
(https://github.com/zblanco/runic), adapted for the Cerebros NAS framework.

Runic models programs as data-driven workflows composed of Steps, Conditions,
Rules, and Workflows that form directed acyclic graphs (DAGs) supporting
lazily evaluated concurrent execution.

Key concepts:
- Step: Basic input -> output transformation
- Condition: Standalone predicate for boolean checks
- Rule: Condition-reaction pair (if condition then reaction)
- Workflow: DAG of connected components with concurrent execution
- Transmutable: Protocol for converting components into executable workflows
"""

from cerebros.runic.component import (
    Component,
    Step,
    Condition,
    Rule,
    Pipeline,
    Transmutable,
)
from cerebros.runic.workflow import Workflow, WorkflowResult
from cerebros.runic.closure import Closure, ClosureMetadata

__all__ = [
    "Component",
    "Step",
    "Condition",
    "Rule",
    "Pipeline",
    "Transmutable",
    "Workflow",
    "WorkflowResult",
    "Closure",
    "ClosureMetadata",
]
