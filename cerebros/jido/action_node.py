"""ActionNode: Wraps domain actions as Runic workflow nodes.

Converts Jido Actions into Runic-compatible workflow components with
automatic schema introspection. This bridges the gap between Jido's
action-based agent system and Runic's DAG workflow execution.

Analogous to JidoRunic's ActionNode module.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from cerebros.runic.component import Component, ComponentStatus, Port, Step
from cerebros.jido.signal import Signal, SignalType
from cerebros.jido.fact import Fact, SignalFact


@dataclass
class ActionSchema:
    """Schema definition for an ActionNode's inputs and outputs.

    Provides automatic introspection of the action's function signature
    to determine required parameters and their types.
    """
    params: Dict[str, type] = field(default_factory=dict)
    returns: Optional[type] = None

    @classmethod
    def from_callable(cls, func: Callable) -> "ActionSchema":
        """Introspect a callable to build its schema."""
        sig = inspect.signature(func)
        params = {}
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            ann = param.annotation
            params[name] = ann if ann != inspect.Parameter.empty else object
        ret = sig.return_annotation
        returns = ret if ret != inspect.Signature.empty else None
        return cls(params=params, returns=returns)


@dataclass
class ActionNode(Component):
    """A workflow node that wraps a domain action.

    ActionNodes bridge Jido actions with Runic workflows by:
    1. Wrapping callables as workflow-compatible components
    2. Auto-introspecting function schemas for port contracts
    3. Emitting signals on execution (start, complete, error)
    4. Producing Facts from execution results

    This is the primary integration point between Jido's action system
    and Runic's DAG execution engine.

    Examples:
        >>> def train_model(spec, data, epochs=10):
        ...     model = build_model(spec)
        ...     return model.fit(data, epochs=epochs)
        >>> node = ActionNode(name="train", action=train_model)
        >>> node.schema.params  # Auto-introspected
        {'spec': <class 'object'>, 'data': <class 'object'>, 'epochs': <class 'object'>}
    """
    action: Optional[Callable] = None
    schema: Optional[ActionSchema] = None
    emit_signals: bool = True
    _signals_emitted: List[Signal] = field(default_factory=list)
    _facts_produced: List[Fact] = field(default_factory=list)

    def __post_init__(self):
        if self.action and self.schema is None:
            self.schema = ActionSchema.from_callable(self.action)
        if self.schema:
            self.input_ports = [
                Port(name=k, port_type=v)
                for k, v in self.schema.params.items()
            ]
            if self.schema.returns:
                self.output_ports = [
                    Port(name="result", port_type=self.schema.returns)
                ]

    def _emit(self, signal: Signal):
        self._signals_emitted.append(signal)

    def prepare(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.status = ComponentStatus.PREPARING
        self._signals_emitted = []
        self._facts_produced = []

        if self.emit_signals:
            self._emit(Signal(
                type=SignalType.COMMAND,
                data={"action": self.name, "context_keys": list(context.keys())},
                source=self.name,
            ))

        if self.schema:
            for param_name in self.schema.params:
                port = next(
                    (p for p in self.input_ports if p.name == param_name), None
                )
                if port and port.required and param_name not in context:
                    pass  # Allow execution to proceed; action may have defaults

        return context

    def execute(self, context: Dict[str, Any]) -> Any:
        self.status = ComponentStatus.EXECUTING

        if self.emit_signals:
            self._emit(Signal(
                type=SignalType.EVENT,
                data={"action": self.name, "phase": "execute_start"},
                source=self.name,
            ))

        if self.action is None:
            return None

        sig = inspect.signature(self.action)
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            if param_name in context:
                kwargs[param_name] = context[param_name]
            elif param.default != inspect.Parameter.empty:
                kwargs[param_name] = param.default

        result = self.action(**kwargs)

        if self.emit_signals:
            result_signal = Signal(
                type=SignalType.RESULT,
                data=result,
                source=self.name,
            )
            self._emit(result_signal)

            fact = SignalFact.signal_to_fact(
                result_signal,
                value=result,
                producer_id=self.name,
            )
            self._facts_produced.append(fact)

        return result

    @property
    def signals(self) -> List[Signal]:
        return list(self._signals_emitted)

    @property
    def facts(self) -> List[Fact]:
        return list(self._facts_produced)
