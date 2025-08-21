"""
Utilities to export a Keras Functional model into a neutral JSON graph spec
that can be used to rebuild the model in other frameworks (e.g., PyTorch).

Spec format (v0):
{
  "version": 0,
  "inputs": ["<input_tensor_name>", ...],
  "outputs": ["<output_tensor_name>", ...],
  "nodes": [
    {
      "name": "<layer_name>",
      "class_name": "Dense|BatchNormalization|Dropout|Add|Concatenate|InputLayer|...",
      "config": { ...layer.get_config()... },
      "inputs": ["<upstream_layer_name>", ...]
    },
    ...
  ]
}
"""
from __future__ import annotations

from typing import Any, Dict, List
import json

try:
    import tensorflow as tf  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    tf = None  # type: ignore


def _safe_layer_config(layer) -> Dict[str, Any]:
    try:
        return layer.get_config() or {}
    except Exception:
        return {}


def _tensor_name(t) -> str:
    try:
        # Tensor names look like 'dense/BiasAdd:0' — keep the op name
        return str(t.name).split(":")[0]
    except Exception:
        return str(t)


def _history_layer_of_tensor(t):
    """Return producing layer for a Keras/TensorFlow tensor if available."""
    for attr in ("_keras_history", "keras_history"):
        if hasattr(t, attr):
            hist = getattr(t, attr)
            # Keras 3 may expose .layer; Keras 2 exposes tuple (layer, node_index, tensor_index)
            layer = getattr(hist, "layer", None)
            if layer is not None:
                return layer
            try:
                return hist[0]
            except Exception:
                return None
    return None


def _upstream_layer_name_from_tensor(t) -> str:
    layer = _history_layer_of_tensor(t)
    return getattr(layer, "name", _tensor_name(t))


def _flatten_inputs(obj):
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_flatten_inputs(x))
        return out
    return [obj]


def export_keras_to_graph(model) -> Dict[str, Any]:
    """Export a tf.keras.Model to a minimal DAG spec.

    Requires a Functional or Model graph (not a pure Sequential with no names).
    Preserves multiplicity and order of inbound connections.
    """
    if tf is None:
        raise RuntimeError("TensorFlow not available; cannot export Keras model.")

    nodes: List[Dict[str, Any]] = []

    # Build list of nodes with inbound connections (layer order is already topological)
    for layer in model.layers:
        inputs: List[str] = []
        # Prefer deriving from the actual input tensors to preserve duplicates and order
        used_fallback = False
        try:
            # Try node-based tensors first if available to reflect graph edges
            inbound_nodes = getattr(layer, "_inbound_nodes", [])
            for node in inbound_nodes:
                kin = getattr(node, "keras_inputs", None) or getattr(node, "input_tensors", None)
                if kin is not None:
                    for t in _flatten_inputs(kin):
                        inputs.append(_upstream_layer_name_from_tensor(t))
                else:
                    in_layers = getattr(node, "inbound_layers", [])
                    if not isinstance(in_layers, (list, tuple)):
                        in_layers = [in_layers]
                    for in_l in in_layers:
                        if in_l is None:
                            continue
                        inputs.append(in_l.name)
        except Exception:
            used_fallback = True
        
        if used_fallback or not inputs:
            try:
                tensors = getattr(layer, "input", None)
                for t in _flatten_inputs(tensors):
                    inputs.append(_upstream_layer_name_from_tensor(t))
            except Exception:
                pass

        nodes.append(
            {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "config": _safe_layer_config(layer),
                "inputs": inputs,
            }
        )

    inputs = []
    try:
        inputs = [str(getattr(t, "name", _tensor_name(t))).split(":")[0] for t in model.inputs]
    except Exception:
        pass

    outputs = []
    try:
        outputs = [str(getattr(t, "name", _tensor_name(t))).split(":")[0] for t in model.outputs]
    except Exception:
        pass

    spec: Dict[str, Any] = {
        "version": 0,
        "framework": "keras",
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
    }
    return spec


def save_graph_json(spec: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, sort_keys=False)


def export_and_save(model, path: str) -> None:
    spec = export_keras_to_graph(model)
    save_graph_json(spec, path)

