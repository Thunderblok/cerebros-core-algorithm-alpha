"""NAS Action nodes: Cerebros operations wrapped as Runic-Jido ActionNodes.

Each action wraps a core Cerebros NAS operation (generate architecture,
build model, train, evaluate) as a Jido ActionNode that can participate
in Runic DAG workflows with signal emission and fact production.
"""

import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from cerebros.jido.action_node import ActionNode
from cerebros.jido.signal import Signal, SignalType


def _generate_architecture_spec(
    minimum_levels: int,
    maximum_levels: int,
    minimum_units_per_level: int,
    maximum_units_per_level: int,
    minimum_neurons_per_unit: int,
    maximum_neurons_per_unit: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a random neural network architecture specification.

    Produces a spec dict compatible with Cerebros NeuralNetworkFuture,
    selecting random values for levels, units per level, and neurons
    per unit within the given ranges.

    Returns:
        Dict with architecture spec and metadata.
    """
    if seed is not None:
        np.random.seed(seed)

    num_levels = np.random.randint(minimum_levels, maximum_levels + 1)
    spec = {}
    total_params_estimate = 0

    for level_idx in range(1, num_levels + 1):
        num_units = np.random.randint(
            minimum_units_per_level, maximum_units_per_level + 1
        )
        level_units = []
        for _ in range(num_units):
            num_neurons = np.random.randint(
                minimum_neurons_per_unit, maximum_neurons_per_unit + 1
            )
            level_units.append(num_neurons)
            total_params_estimate += num_neurons
        spec[str(level_idx)] = level_units

    return {
        "spec": spec,
        "num_levels": num_levels,
        "total_units": sum(len(v) for v in spec.values()),
        "total_params_estimate": total_params_estimate,
    }


def _evaluate_model_results(
    history: Dict[str, List[float]],
    metric_to_rank_by: str,
    direction: str,
) -> Dict[str, Any]:
    """Evaluate training results and rank the model.

    Args:
        history: Training history dict (metric_name -> list of values).
        metric_to_rank_by: Which metric to use for ranking.
        direction: 'max' or 'min'.

    Returns:
        Evaluation results with best metric value and epoch.
    """
    if metric_to_rank_by not in history:
        return {
            "best_value": None,
            "best_epoch": None,
            "metric": metric_to_rank_by,
            "direction": direction,
            "all_values": history,
        }

    values = history[metric_to_rank_by]
    if direction == "max":
        best_idx = int(np.argmax(values))
    else:
        best_idx = int(np.argmin(values))

    return {
        "best_value": values[best_idx],
        "best_epoch": best_idx,
        "metric": metric_to_rank_by,
        "direction": direction,
        "final_value": values[-1] if values else None,
        "all_values": {k: v for k, v in history.items()},
    }


def GenerateArchitectureAction(
    name: str = "generate_architecture",
    **defaults,
) -> ActionNode:
    """Create an ActionNode that generates random architecture specs.

    This wraps the Cerebros architecture generation logic as a Jido
    ActionNode, emitting ARCHITECTURE_GENERATED signals upon completion.
    """
    def action(
        minimum_levels: int = 2,
        maximum_levels: int = 10,
        minimum_units_per_level: int = 1,
        maximum_units_per_level: int = 5,
        minimum_neurons_per_unit: int = 4,
        maximum_neurons_per_unit: int = 64,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        params = {
            "minimum_levels": minimum_levels,
            "maximum_levels": maximum_levels,
            "minimum_units_per_level": minimum_units_per_level,
            "maximum_units_per_level": maximum_units_per_level,
            "minimum_neurons_per_unit": minimum_neurons_per_unit,
            "maximum_neurons_per_unit": maximum_neurons_per_unit,
            "seed": seed,
        }
        params.update(defaults)
        return _generate_architecture_spec(**params)

    return ActionNode(name=name, action=action, emit_signals=True)


def BuildModelAction(name: str = "build_model") -> ActionNode:
    """Create an ActionNode that builds a Cerebros neural network from a spec.

    This is a placeholder ActionNode that wraps the NeuralNetworkFuture
    materialization step. The actual TensorFlow model building requires
    the full Cerebros context (input shapes, etc), so this node
    coordinates the build workflow and emits MODEL_BUILT signals.
    """
    def action(
        architecture_spec: Optional[Dict] = None,
        generate_architecture: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        spec = architecture_spec or generate_architecture
        if spec is None:
            return {"status": "error", "message": "No architecture spec provided"}

        raw_spec = spec.get("spec", spec)
        num_levels = len(raw_spec) if isinstance(raw_spec, dict) else 0

        return {
            "status": "ready",
            "spec": raw_spec,
            "num_levels": num_levels,
            "total_units": sum(
                len(v) if isinstance(v, list) else 1
                for v in raw_spec.values()
            ) if isinstance(raw_spec, dict) else 0,
            "message": f"Model blueprint ready with {num_levels} levels",
        }

    return ActionNode(name=name, action=action, emit_signals=True)


def TrainModelAction(name: str = "train_model") -> ActionNode:
    """Create an ActionNode for model training coordination.

    Wraps the training step as an ActionNode that emits
    TRAINING_STARTED and TRAINING_COMPLETED signals. The actual
    TensorFlow training is delegated to the Cerebros framework.
    """
    def action(
        build_model: Optional[Dict] = None,
        epochs: int = 7,
        batch_size: int = 200,
        learning_rate: float = 0.005,
        **kwargs,
    ) -> Dict[str, Any]:
        model_info = build_model or {}
        spec = model_info.get("spec", {})

        return {
            "status": "configured",
            "spec": spec,
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            "message": f"Training configured: {epochs} epochs, "
                       f"batch_size={batch_size}, lr={learning_rate}",
        }

    return ActionNode(name=name, action=action, emit_signals=True)


def EvaluateModelAction(name: str = "evaluate_model") -> ActionNode:
    """Create an ActionNode for model evaluation.

    Wraps the evaluation step, computing metrics and ranking models.
    Emits EVALUATION_COMPLETED signals with the evaluation results.
    """
    def action(
        train_model: Optional[Dict] = None,
        history: Optional[Dict] = None,
        metric_to_rank_by: str = "loss",
        direction: str = "min",
        **kwargs,
    ) -> Dict[str, Any]:
        training_info = train_model or {}
        training_history = history or {}

        if not training_history and "training_config" in training_info:
            return {
                "status": "configured",
                "training_info": training_info,
                "metric_to_rank_by": metric_to_rank_by,
                "direction": direction,
                "message": "Evaluation configured, awaiting training results",
            }

        return _evaluate_model_results(
            training_history, metric_to_rank_by, direction
        )

    return ActionNode(name=name, action=action, emit_signals=True)
