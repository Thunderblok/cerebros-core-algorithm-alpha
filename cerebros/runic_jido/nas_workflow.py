"""NAS Workflow: Express the Cerebros NAS pipeline as a Runic DAG.

Builds a Runic Workflow DAG that models the Cerebros Neural Architecture
Search pipeline: generate -> build -> train -> evaluate, with support
for conditional rules, parallel trials, and signal-driven orchestration.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cerebros.runic.component import Condition, Rule, Step
from cerebros.runic.workflow import Workflow
from cerebros.jido.action_node import ActionNode
from cerebros.runic_jido.nas_actions import (
    GenerateArchitectureAction,
    BuildModelAction,
    TrainModelAction,
    EvaluateModelAction,
)


@dataclass
class NASWorkflowBuilder:
    """Builder for constructing NAS pipeline workflows.

    Provides a fluent API for building Runic DAG workflows that model
    the Cerebros NAS pipeline. Supports customization of each stage
    and addition of conditional rules.

    Examples:
        >>> builder = NASWorkflowBuilder(name="cifar10_search")
        >>> builder.with_architecture_params(minimum_levels=3, maximum_levels=15)
        >>> builder.with_training_params(epochs=10, batch_size=128)
        >>> builder.with_evaluation_metric("val_accuracy", direction="max")
        >>> workflow = builder.build()
        >>> result = workflow.run(initial_context)
    """
    name: str = "nas_workflow"
    architecture_params: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    evaluation_metric: str = "loss"
    evaluation_direction: str = "min"
    custom_steps: List[tuple] = field(default_factory=list)
    custom_rules: List[tuple] = field(default_factory=list)
    pre_build_hooks: List[Callable] = field(default_factory=list)
    post_train_hooks: List[Callable] = field(default_factory=list)

    def with_architecture_params(self, **params) -> "NASWorkflowBuilder":
        """Configure architecture generation parameters."""
        self.architecture_params.update(params)
        return self

    def with_training_params(self, **params) -> "NASWorkflowBuilder":
        """Configure training parameters."""
        self.training_params.update(params)
        return self

    def with_evaluation_metric(
        self, metric: str, direction: str = "min"
    ) -> "NASWorkflowBuilder":
        """Set the evaluation metric and optimization direction."""
        self.evaluation_metric = metric
        self.evaluation_direction = direction
        return self

    def add_step(
        self, step: Any, depends_on: Optional[List[str]] = None
    ) -> "NASWorkflowBuilder":
        """Add a custom step to the workflow."""
        self.custom_steps.append((step, depends_on))
        return self

    def add_rule(
        self, rule: Rule, depends_on: Optional[List[str]] = None
    ) -> "NASWorkflowBuilder":
        """Add a conditional rule to the workflow."""
        self.custom_rules.append((rule, depends_on))
        return self

    def add_pre_build_hook(self, hook: Callable) -> "NASWorkflowBuilder":
        """Add a hook to run before model building."""
        self.pre_build_hooks.append(hook)
        return self

    def add_post_train_hook(self, hook: Callable) -> "NASWorkflowBuilder":
        """Add a hook to run after training."""
        self.post_train_hooks.append(hook)
        return self

    def build(self) -> Workflow:
        """Build the NAS workflow DAG.

        Creates a Runic Workflow with the following DAG structure:
            generate_architecture -> build_model -> train_model -> evaluate_model

        With optional custom steps and rules inserted at appropriate points.
        """
        wf = Workflow(name=self.name)

        # Core NAS pipeline nodes
        gen_node = GenerateArchitectureAction(
            name="generate_architecture",
            **self.architecture_params,
        )
        build_node = BuildModelAction(name="build_model")
        train_node = TrainModelAction(name="train_model")
        eval_node = EvaluateModelAction(name="evaluate_model")

        # Add core pipeline
        wf.add_component(gen_node)
        wf.add_component(build_node, depends_on=["generate_architecture"])

        # Insert pre-build hooks as steps
        last_pre_build = "build_model"
        for i, hook in enumerate(self.pre_build_hooks):
            hook_step = Step(
                name=f"pre_build_hook_{i}",
                func=hook,
            )
            wf.add_component(hook_step, depends_on=["build_model"])
            last_pre_build = hook_step.name

        wf.add_component(
            train_node,
            depends_on=[last_pre_build] if self.pre_build_hooks else ["build_model"],
        )

        # Insert post-train hooks
        last_post_train = "train_model"
        for i, hook in enumerate(self.post_train_hooks):
            hook_step = Step(
                name=f"post_train_hook_{i}",
                func=hook,
            )
            wf.add_component(hook_step, depends_on=["train_model"])
            last_post_train = hook_step.name

        wf.add_component(
            eval_node,
            depends_on=[last_post_train] if self.post_train_hooks else ["train_model"],
        )

        # Add custom steps
        for step, deps in self.custom_steps:
            wf.add_component(step, depends_on=deps)

        # Add custom rules
        for rule, deps in self.custom_rules:
            wf.add_component(rule, depends_on=deps)

        # Inject default context values for evaluation
        eval_defaults = Step(
            name="eval_defaults",
            func=lambda ctx: {
                "metric_to_rank_by": self.evaluation_metric,
                "direction": self.evaluation_direction,
            },
        )
        wf.add_component(eval_defaults)

        return wf


def create_nas_workflow(
    name: str = "nas_pipeline",
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
) -> Workflow:
    """Convenience function to create a standard NAS workflow.

    Creates a fully configured Runic DAG workflow for Neural Architecture
    Search with sensible defaults matching the Cerebros framework.

    Returns:
        A ready-to-run Workflow instance.
    """
    builder = NASWorkflowBuilder(name=name)
    builder.with_architecture_params(
        minimum_levels=minimum_levels,
        maximum_levels=maximum_levels,
        minimum_units_per_level=minimum_units_per_level,
        maximum_units_per_level=maximum_units_per_level,
        minimum_neurons_per_unit=minimum_neurons_per_unit,
        maximum_neurons_per_unit=maximum_neurons_per_unit,
    )
    builder.with_training_params(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    builder.with_evaluation_metric(metric_to_rank_by, direction)
    return builder.build()
