"""Runic workflow components: Step, Condition, Rule, Pipeline.

These are the building blocks of Runic workflows, ported from the Elixir
Runic library's component system. Each component implements the Transmutable
protocol for conversion into executable workflow nodes.

Components connect together in a Workflow supporting lazily evaluated
concurrent execution, forming decorated dataflow graphs (DAGs).
"""

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from cerebros.runic.closure import Closure


class ComponentStatus(Enum):
    """Execution status of a workflow component."""
    PENDING = "pending"
    PREPARING = "preparing"
    EXECUTING = "executing"
    APPLYING = "applying"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Transmutable(ABC):
    """Protocol for converting components into executable workflow nodes.

    Analogous to Runic's Transmutable protocol which converts components
    (Steps, Rules, etc.) into evaluable Workflow structs via `transmute/1`.
    """

    @abstractmethod
    def transmute(self) -> "Component":
        """Convert this object into an executable Component."""
        ...


@dataclass
class Port:
    """Input or output port declaration for type-driven composition.

    Analogous to Runic's port contracts that declare input_ports and
    output_ports for boundary validation.
    """
    name: str
    port_type: type = object
    required: bool = True
    default: Any = None


@dataclass
class Component(ABC):
    """Base class for all Runic workflow components.

    Implements the three-phase execution model from Runic:
    1. prepare: Validate inputs and set up execution context
    2. execute: Perform the core computation
    3. apply: Apply results to the workflow state

    Attributes:
        id: Unique identifier for this component instance.
        name: Human-readable name.
        status: Current execution status.
        input_ports: Declared input port contracts.
        output_ports: Declared output port contracts.
        dependencies: Set of component IDs this depends on.
        result: Stored result after execution.
        error: Stored error if execution failed.
    """
    name: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: ComponentStatus = ComponentStatus.PENDING
    input_ports: List[Port] = field(default_factory=list)
    output_ports: List[Port] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    result: Any = None
    error: Optional[Exception] = None

    @property
    def content_hash(self) -> str:
        raw = f"{self.__class__.__name__}:{self.name}:{self.id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def prepare(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Validate inputs and prepare execution context."""
        self.status = ComponentStatus.PREPARING
        return context

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """Phase 2: Perform the core computation."""
        ...

    def apply(self, result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Apply results to workflow state."""
        self.status = ComponentStatus.APPLYING
        self.result = result
        context[self.id] = result
        if self.name:
            context[self.name] = result
        self.status = ComponentStatus.COMPLETED
        return context

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all three phases in sequence."""
        try:
            context = self.prepare(context)
            result = self.execute(context)
            context = self.apply(result, context)
        except Exception as e:
            self.status = ComponentStatus.FAILED
            self.error = e
            raise
        return context


@dataclass
class Step(Component):
    """A basic input->output transformation step.

    The simplest Runic component: takes input from the workflow context,
    applies a function, and produces output. Supports 0, 1, or 2-arity
    functions.

    Examples:
        >>> step = Step(name="double", func=lambda x: x * 2, input_key="value")
        >>> step.run({"value": 5})
        {'value': 5, 'double': 10, ...}
    """
    func: Optional[Callable] = None
    input_key: Optional[str] = None
    input_keys: Optional[List[str]] = None

    def __post_init__(self):
        if self.func and not isinstance(self.func, Closure):
            self.func = Closure(self.func, name=self.name or "step")

    def execute(self, context: Dict[str, Any]) -> Any:
        self.status = ComponentStatus.EXECUTING
        if self.func is None:
            return context

        if self.input_keys:
            args = [context.get(k) for k in self.input_keys]
            return self.func(*args)
        elif self.input_key:
            return self.func(context.get(self.input_key))
        else:
            return self.func(context)


@dataclass
class Condition(Component):
    """A standalone predicate for boolean evaluation.

    Can be reused across multiple Rules. Returns True/False based on
    the workflow context. Analogous to Runic's Condition component.

    Examples:
        >>> cond = Condition(name="is_positive", predicate=lambda x: x > 0, input_key="value")
        >>> cond.execute({"value": 5})
        True
    """
    predicate: Optional[Callable] = None
    input_key: Optional[str] = None

    def __post_init__(self):
        if self.predicate and not isinstance(self.predicate, Closure):
            self.predicate = Closure(self.predicate, name=self.name or "condition")

    def execute(self, context: Dict[str, Any]) -> bool:
        self.status = ComponentStatus.EXECUTING
        if self.predicate is None:
            return True
        if self.input_key:
            return bool(self.predicate(context.get(self.input_key)))
        return bool(self.predicate(context))


@dataclass
class Rule(Component):
    """A condition-reaction pair: if condition holds, execute reaction.

    Combines a Condition with a reaction Step. When the condition evaluates
    to True, the reaction is executed. Analogous to Runic's Rule with
    condition:/reaction: keyword form.

    Examples:
        >>> rule = Rule(
        ...     name="scale_if_large",
        ...     condition=Condition(predicate=lambda ctx: ctx.get("size", 0) > 100),
        ...     reaction=Step(func=lambda ctx: ctx["size"] * 0.5),
        ... )
    """
    condition: Optional[Condition] = None
    reaction: Optional[Step] = None

    def execute(self, context: Dict[str, Any]) -> Any:
        self.status = ComponentStatus.EXECUTING
        if self.condition is None or self.condition.execute(context):
            if self.reaction:
                return self.reaction.execute(context)
            return None
        self.status = ComponentStatus.SKIPPED
        return None

    def apply(self, result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.status == ComponentStatus.SKIPPED:
            self.status = ComponentStatus.COMPLETED
            return context
        return super().apply(result, context)


@dataclass
class Pipeline(Component):
    """A sequential chain of Steps executed in order.

    Each step's output becomes available in the context for subsequent steps.
    Analogous to composing Runic steps with the pipeline syntax.
    """
    steps: List[Component] = field(default_factory=list)

    def execute(self, context: Dict[str, Any]) -> Any:
        self.status = ComponentStatus.EXECUTING
        current_context = dict(context)
        last_result = None
        for step in self.steps:
            current_context = step.run(current_context)
            last_result = step.result
        return last_result

    def apply(self, result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        context = super().apply(result, context)
        for step in self.steps:
            if step.result is not None:
                context[step.id] = step.result
                if step.name:
                    context[step.name] = step.result
        return context
