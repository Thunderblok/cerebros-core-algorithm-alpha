"""Cerebros Runic-Jido Bridge: Connecting DAG workflows with signal-driven agents.

This package bridges the Runic workflow engine with Jido's agent system,
enabling Cerebros NAS pipelines to be expressed as signal-reactive DAG
workflows with full execution tracing and provenance tracking.

Inspired by JidoRunic (https://github.com/agentjido/jido_runic) which
bridges Runic's DAG workflow composition with Jido's signal-driven
agent architecture in Elixir.

Key integration points:
- NASWorkflow: Expresses the Cerebros NAS pipeline as a Runic DAG
- NASAgent: Signal-driven agent that orchestrates architecture search
- ActionNode wrappers for Cerebros operations (generate, build, train, evaluate)
- Signal types for NAS lifecycle events
"""

from cerebros.runic_jido.nas_workflow import (
    NASWorkflowBuilder,
    create_nas_workflow,
)
from cerebros.runic_jido.nas_agent import NASAgent
from cerebros.runic_jido.nas_actions import (
    GenerateArchitectureAction,
    BuildModelAction,
    TrainModelAction,
    EvaluateModelAction,
)

__all__ = [
    "NASWorkflowBuilder",
    "create_nas_workflow",
    "NASAgent",
    "GenerateArchitectureAction",
    "BuildModelAction",
    "TrainModelAction",
    "EvaluateModelAction",
]
