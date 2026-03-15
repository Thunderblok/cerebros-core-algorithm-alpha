"""NAS Agent: Signal-driven agent for Neural Architecture Search.

Combines the Runic DAG workflow with Jido's agent loop to create a
fully reactive NAS orchestrator. The NASAgent receives signals (e.g.,
"start search", "architecture generated", "training complete") and
routes them through appropriate strategies.

This is the top-level integration of Runic workflow composition with
Jido signal-driven agent architecture for the Cerebros NAS framework.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from cerebros.jido.signal import Signal, SignalType
from cerebros.jido.fact import Fact, SignalFact
from cerebros.jido.strategy import AgentLoop, Strategy, SignalMatcher
from cerebros.jido.introspection import Introspection
from cerebros.runic.workflow import Workflow
from cerebros.runic_jido.nas_workflow import NASWorkflowBuilder, create_nas_workflow


@dataclass
class NASAgent:
    """Signal-driven agent for orchestrating Neural Architecture Search.

    Combines Runic's DAG workflow engine with Jido's agent loop to
    create a reactive NAS system. The agent:

    1. Receives COMMAND signals to initiate search
    2. Routes signals through NAS strategies (generate, build, train, eval)
    3. Emits lifecycle signals (ARCHITECTURE_GENERATED, MODEL_BUILT, etc.)
    4. Collects Facts with full provenance tracking
    5. Supports introspection of execution history

    Examples:
        >>> agent = NASAgent.create(
        ...     name="cifar10_nas",
        ...     minimum_levels=3,
        ...     maximum_levels=15,
        ...     epochs=10,
        ... )
        >>> results = agent.run_search(num_trials=5)
        >>> print(agent.introspection.execution_stats())
    """
    name: str = "nas_agent"
    agent_loop: AgentLoop = field(default_factory=lambda: AgentLoop(name="nas"))
    introspection: Optional[Introspection] = None
    search_config: Dict[str, Any] = field(default_factory=dict)
    _trial_results: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.introspection is None:
            self.introspection = Introspection(self.agent_loop)

    @classmethod
    def create(
        cls,
        name: str = "nas_agent",
        minimum_levels: int = 2,
        maximum_levels: int = 10,
        minimum_units_per_level: int = 1,
        maximum_units_per_level: int = 5,
        minimum_neurons_per_unit: int = 4,
        maximum_neurons_per_unit: int = 64,
        epochs: int = 7,
        batch_size: int = 200,
        learning_rate: float = 0.005,
        metric_to_rank_by: str = "loss",
        direction: str = "min",
    ) -> "NASAgent":
        """Factory method to create a fully configured NASAgent.

        Creates the agent with a NAS workflow strategy that handles
        COMMAND signals and routes them through the full NAS pipeline.
        """
        workflow = create_nas_workflow(
            name=f"{name}_workflow",
            minimum_levels=minimum_levels,
            maximum_levels=maximum_levels,
            minimum_units_per_level=minimum_units_per_level,
            maximum_units_per_level=maximum_units_per_level,
            minimum_neurons_per_unit=minimum_neurons_per_unit,
            maximum_neurons_per_unit=maximum_neurons_per_unit,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            metric_to_rank_by=metric_to_rank_by,
            direction=direction,
        )

        nas_strategy = Strategy(
            name="nas_search",
            workflow=workflow,
            signal_matcher=SignalMatcher(
                patterns={SignalType.COMMAND, SignalType.EVENT}
            ),
        )

        loop = AgentLoop(name=name)
        loop.register_strategy(nas_strategy)

        agent = cls(
            name=name,
            agent_loop=loop,
            search_config={
                "minimum_levels": minimum_levels,
                "maximum_levels": maximum_levels,
                "minimum_units_per_level": minimum_units_per_level,
                "maximum_units_per_level": maximum_units_per_level,
                "minimum_neurons_per_unit": minimum_neurons_per_unit,
                "maximum_neurons_per_unit": maximum_neurons_per_unit,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "metric_to_rank_by": metric_to_rank_by,
                "direction": direction,
            },
        )

        return agent

    def run_trial(
        self,
        trial_number: int = 0,
        seed: Optional[int] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[Fact]:
        """Run a single NAS trial through the agent loop.

        Emits a COMMAND signal that triggers the NAS workflow,
        generating an architecture, building, training, and evaluating.

        Args:
            trial_number: Trial identifier.
            seed: Random seed for architecture generation.
            extra_context: Additional context to pass to the workflow.

        Returns:
            List of Facts produced by the trial.
        """
        signal_data = {
            "trial_number": trial_number,
            "action": "run_nas_trial",
            **self.search_config,
        }
        if seed is not None:
            signal_data["seed"] = seed
        if extra_context:
            signal_data.update(extra_context)

        signal = Signal(
            type=SignalType.COMMAND,
            data=signal_data,
            source=self.name,
        )

        facts = self.agent_loop.process_signal(signal)

        for fact in facts:
            if isinstance(fact.value, dict):
                self._trial_results.append({
                    "trial_number": trial_number,
                    "fact_id": fact.id,
                    **fact.value,
                })

        return facts

    def run_search(
        self,
        num_trials: int = 5,
        base_seed: int = 8675309,
    ) -> List[Dict[str, Any]]:
        """Run multiple NAS trials, collecting results.

        Args:
            num_trials: Number of architecture trials to run.
            base_seed: Base random seed (incremented per trial).

        Returns:
            List of trial result dicts with architecture specs and metrics.
        """
        self._trial_results = []

        for i in range(num_trials):
            self.run_trial(
                trial_number=i,
                seed=base_seed + i,
            )

            # Emit lifecycle signal
            self.agent_loop.process_signal(Signal(
                type=SignalType.EVENT,
                data={
                    "action": "trial_completed",
                    "trial_number": i,
                    "total_trials": num_trials,
                },
                source=self.name,
            ))

        return list(self._trial_results)

    @property
    def trial_results(self) -> List[Dict[str, Any]]:
        return list(self._trial_results)

    @property
    def execution_stats(self) -> Dict[str, Any]:
        return self.introspection.execution_stats()

    def visualize_workflow(self) -> str:
        """Get a text visualization of the NAS workflow DAG."""
        for strategy in self.agent_loop.strategies:
            if strategy.workflow:
                return strategy.workflow.visualize()
        return "No workflow configured"

    def __repr__(self) -> str:
        return (
            f"NASAgent(name={self.name!r}, "
            f"trials={len(self._trial_results)}, "
            f"signals={len(self.agent_loop.signal_log)}, "
            f"facts={len(self.agent_loop.fact_store)})"
        )
