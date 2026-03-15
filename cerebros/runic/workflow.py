"""Runic Workflow: DAG-based concurrent execution engine.

Implements the core Workflow struct from Runic, which connects components
into a directed acyclic graph supporting lazily evaluated concurrent
execution. Non-linear pipelines allow multiple downstream nodes to consume
outputs from a single upstream node without redundant computation.

The three-phase execution model (prepare, execute, apply) supports concurrent
task processing with configurable concurrency limits.
"""

import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cerebros.runic.component import Component, ComponentStatus


class WorkflowStatus(Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class WorkflowResult:
    """Result of a workflow execution.

    Attributes:
        status: Final workflow status.
        context: The final execution context with all component results.
        execution_order: List of component IDs in the order they executed.
        duration_ms: Total execution time in milliseconds.
        errors: Dict of component_id -> error for any failed components.
        provenance: Execution lineage tracking (component_id -> parent_ids).
    """
    status: WorkflowStatus
    context: Dict[str, Any]
    execution_order: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    errors: Dict[str, Exception] = field(default_factory=dict)
    provenance: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class Workflow:
    """A directed acyclic graph of connected components.

    Components connect via declared dependencies, forming a DAG that
    supports concurrent execution of independent branches. This mirrors
    Runic's Workflow struct where components connect via tuples like
    {parent, [children]}.

    Supports:
    - Non-linear pipelines (fan-out from single nodes)
    - Concurrent execution of independent branches
    - Three-phase execution (prepare, execute, apply)
    - Configurable concurrency limits
    - Runtime workflow modification (adding/removing components)

    Examples:
        >>> wf = Workflow(name="my_pipeline")
        >>> step_a = Step(name="fetch", func=fetch_data)
        >>> step_b = Step(name="process", func=process_data, input_key="fetch")
        >>> wf.add_component(step_a)
        >>> wf.add_component(step_b, depends_on=["fetch"])
        >>> result = wf.run({"url": "..."})
    """
    name: str = "workflow"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    components: Dict[str, Component] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse_edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    max_concurrency: int = 4
    status: WorkflowStatus = WorkflowStatus.PENDING
    _hooks: Dict[str, List[Callable]] = field(default_factory=lambda: defaultdict(list))

    def add_component(
        self,
        component: Component,
        depends_on: Optional[List[str]] = None,
    ) -> "Workflow":
        """Add a component to the workflow DAG.

        Args:
            component: The component to add.
            depends_on: List of component names or IDs this depends on.

        Returns:
            self for chaining.
        """
        key = component.name or component.id
        self.components[key] = component

        if depends_on:
            for dep in depends_on:
                dep_key = self._resolve_key(dep)
                if dep_key:
                    self.edges[dep_key].add(key)
                    self.reverse_edges[key].add(dep_key)
                    component.dependencies.add(dep_key)

        return self

    def connect(self, from_name: str, to_name: str) -> "Workflow":
        """Connect two components: from_name -> to_name (to depends on from)."""
        from_key = self._resolve_key(from_name)
        to_key = self._resolve_key(to_name)
        if from_key and to_key:
            self.edges[from_key].add(to_key)
            self.reverse_edges[to_key].add(from_key)
            self.components[to_key].dependencies.add(from_key)
        return self

    def add_hook(self, event: str, callback: Callable) -> "Workflow":
        """Register a hook callback for workflow events.

        Events: 'before_execute', 'after_execute', 'on_error', 'on_complete'
        """
        self._hooks[event].append(callback)
        return self

    def _resolve_key(self, name_or_id: str) -> Optional[str]:
        if name_or_id in self.components:
            return name_or_id
        for key, comp in self.components.items():
            if comp.id == name_or_id or comp.name == name_or_id:
                return key
        return None

    def _topological_sort(self) -> List[List[str]]:
        """Compute execution layers via topological sort.

        Returns a list of layers, where each layer contains component keys
        that can execute concurrently (all dependencies satisfied).
        """
        in_degree = {}
        for key in self.components:
            in_degree[key] = len(self.reverse_edges.get(key, set()))

        layers = []
        remaining = set(self.components.keys())

        while remaining:
            ready = [k for k in remaining if in_degree.get(k, 0) == 0]
            if not ready:
                raise ValueError(
                    f"Cycle detected in workflow DAG. Remaining: {remaining}"
                )
            layers.append(ready)
            for k in ready:
                remaining.remove(k)
                for child in self.edges.get(k, set()):
                    in_degree[child] -= 1

        return layers

    def _fire_hooks(self, event: str, **kwargs):
        for callback in self._hooks.get(event, []):
            callback(**kwargs)

    def run(
        self,
        initial_context: Optional[Dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
    ) -> WorkflowResult:
        """Execute the workflow DAG.

        Runs components in topological order, executing independent
        components concurrently within each layer.

        Args:
            initial_context: Starting context dict.
            max_concurrency: Override default concurrency limit.

        Returns:
            WorkflowResult with final context and execution metadata.
        """
        start_time = time.time()
        self.status = WorkflowStatus.RUNNING
        context = dict(initial_context or {})
        execution_order = []
        errors = {}
        provenance = {}
        concurrency = max_concurrency or self.max_concurrency

        try:
            layers = self._topological_sort()
        except ValueError as e:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                context=context,
                errors={"_workflow": e},
                duration_ms=(time.time() - start_time) * 1000,
            )

        for layer in layers:
            self._fire_hooks("before_execute", layer=layer, context=context)

            if len(layer) == 1 or concurrency <= 1:
                for key in layer:
                    comp = self.components[key]
                    provenance[key] = list(self.reverse_edges.get(key, set()))
                    try:
                        context = comp.run(context)
                        execution_order.append(key)
                    except Exception as e:
                        errors[key] = e
                        self._fire_hooks("on_error", component=key, error=e)
                        self.status = WorkflowStatus.FAILED
                        return WorkflowResult(
                            status=WorkflowStatus.FAILED,
                            context=context,
                            execution_order=execution_order,
                            errors=errors,
                            provenance=provenance,
                            duration_ms=(time.time() - start_time) * 1000,
                        )
            else:
                with ThreadPoolExecutor(max_workers=min(concurrency, len(layer))) as pool:
                    futures = {}
                    for key in layer:
                        comp = self.components[key]
                        provenance[key] = list(self.reverse_edges.get(key, set()))
                        comp_context = dict(context)
                        futures[pool.submit(comp.run, comp_context)] = key

                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            result_context = future.result()
                            context.update(result_context)
                            execution_order.append(key)
                        except Exception as e:
                            errors[key] = e
                            self._fire_hooks("on_error", component=key, error=e)

                if errors:
                    self.status = WorkflowStatus.FAILED
                    return WorkflowResult(
                        status=WorkflowStatus.FAILED,
                        context=context,
                        execution_order=execution_order,
                        errors=errors,
                        provenance=provenance,
                        duration_ms=(time.time() - start_time) * 1000,
                    )

            self._fire_hooks("after_execute", layer=layer, context=context)

        self.status = WorkflowStatus.COMPLETED
        self._fire_hooks("on_complete", context=context)

        return WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            context=context,
            execution_order=execution_order,
            provenance=provenance,
            duration_ms=(time.time() - start_time) * 1000,
        )

    def visualize(self) -> str:
        """Return a text-based visualization of the workflow DAG."""
        lines = [f"Workflow: {self.name}"]
        lines.append("=" * 40)
        try:
            layers = self._topological_sort()
        except ValueError:
            lines.append("ERROR: Cycle detected in DAG")
            return "\n".join(lines)

        for i, layer in enumerate(layers):
            lines.append(f"Layer {i}:")
            for key in layer:
                comp = self.components[key]
                deps = self.reverse_edges.get(key, set())
                children = self.edges.get(key, set())
                dep_str = f" <- [{', '.join(deps)}]" if deps else ""
                child_str = f" -> [{', '.join(children)}]" if children else ""
                lines.append(
                    f"  [{comp.__class__.__name__}] {key}"
                    f" ({comp.status.value}){dep_str}{child_str}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Workflow(name={self.name!r}, "
            f"components={len(self.components)}, "
            f"status={self.status.value})"
        )
