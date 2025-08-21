"""Minimal end-to-end smoke test (extended) for Cerebros core components.

Steps:
1. Seed RNGs for determinism.
2. Create a trivial Keras model directly (keeps it fast) OR (future) a tiny Cerebros search.
3. Train for 1 epoch on synthetic regression data.
4. Save (.keras + meta) and reload model.
5. Export to graph JSON.
6. Print a short success summary.

This script is intentionally lightweight so it can be run locally or in CI
as a quick “is the plumbing intact?” check without invoking the full search
machinery. We can extend later to instantiate a minimal SimpleCerebrosRandomSearch
if desired, but direct model creation keeps runtime < a few seconds on CPU.
"""
from __future__ import annotations

import os, json, tempfile, time

# Force CPU for stability in lightweight smoke test (avoid GPU PTX mismatches)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
try:  # Prefer standalone keras if available (Keras 3)
    import keras
except Exception:  # fallback to tf.keras namespace
    from tensorflow import keras  # type: ignore

from cerebros.utils.random import set_global_seed
from cerebros.persistence.keras_io import save_model_and_meta, load_model_safe
from cerebros.keras_export import export_and_save

# Cerebros search components (for extended mode)
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search import (
    SimpleCerebrosRandomSearch,
)
from cerebros.units.units import Unit  # type: ignore
from multiprocessing import Lock
import argparse


def build_tiny_model(input_dim: int = 8) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="smoke_input")
    x = keras.layers.Dense(16, activation="elu", name="dense_1")(inputs)
    x = keras.layers.Dense(8, activation="elu", name="dense_2")(x)
    outputs = keras.layers.Dense(1, activation=None, name="regression_head")(x)
    model = keras.Model(inputs, outputs, name="smoke_model")
    # Type ignore: static analyzer may not recognize optimizer object signature
    model.compile(optimizer=keras.optimizers.Adam(0.005), loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])  # type: ignore[arg-type]
    return model


def run_basic_keras_smoke(seed: int):
    model = build_tiny_model()
    # Synthetic regression data
    n = 256
    x = np.random.randn(n, 8).astype("float32")
    y = (x.sum(axis=1, keepdims=True) + np.random.randn(n, 1) * 0.05).astype("float32")

    history = model.fit(x, y, epochs=1, batch_size=32, verbose=0)  # type: ignore[arg-type]

    tmpdir = tempfile.mkdtemp(prefix="cerebros_smoke_")
    save_path = os.path.join(tmpdir, "smoke_model.keras")
    meta = {
        "seed": seed,
        "epochs": 1,
        "loss_history": history.history.get("loss", []),
        "mode": "keras",
    }
    save_model_and_meta(model, save_path, meta)

    reloaded = load_model_safe(save_path)
    preds_orig = model.predict(x[:8], verbose=0)  # type: ignore[arg-type]
    preds_reload = reloaded.predict(x[:8], verbose=0)  # type: ignore[arg-type]
    max_abs_diff = float(np.max(np.abs(preds_orig - preds_reload)))
    if max_abs_diff > 1e-5:
        raise AssertionError(f"Reloaded model predictions diverged: max_abs_diff={max_abs_diff}")

    graph_path = os.path.join(tmpdir, "smoke_model.graph.json")
    export_and_save(model, graph_path)
    return {
        "tmpdir": tmpdir,
        "save_path": save_path,
        "graph_path": graph_path,
        "nodes": 4,
        "max_abs_diff": max_abs_diff,
    }


def run_cerebros_smoke(seed: int,
                       levels: int,
                       units_per_level: int,
                       neurons_min: int,
                       neurons_max: int,
                       epochs: int,
                       batch_size: int) -> dict:
    # Synthetic data (slightly larger to exercise graph)
    n = 512
    in_dim = 16
    x = np.random.randn(n, in_dim).astype("float32")
    # Non-trivial target: linear + non-linear mix
    y = (2 * x[:, :4].sum(axis=1) + np.sin(x[:, 4:8]).sum(axis=1) + np.random.randn(n) * 0.1).astype("float32")
    y = y.reshape(-1, 1)

    scrs = SimpleCerebrosRandomSearch(
        unit_type=Unit,  # type: ignore[arg-type]
        input_shapes=[(in_dim,)],
        output_shapes=[1],
        training_data=[x],
        labels=[y],
        validation_split=0.1,
        direction="minimize",
        metric_to_rank_by="loss",
        minimum_levels=levels,
        maximum_levels=levels,
        minimum_units_per_level=units_per_level,
        maximum_units_per_level=units_per_level,
        minimum_neurons_per_unit=neurons_min,
        maximum_neurons_per_unit=neurons_max,
        activation="elu",
        final_activation=None,
        number_of_architecture_moities_to_try=1,
        number_of_tries_per_architecture_moity=1,
        minimum_skip_connection_depth=1,
        maximum_skip_connection_depth=min(7, levels),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        project_name="cerebros-smoke-project",
        verify_persistence=True,
        persistence_tolerance=1e-5,
    )

    # Build a single spec deterministically (internally random but constrained)
    scrs.parse_neural_network_structural_spec_random()
    spec = scrs.get_neural_network_spec()
    lock = Lock()
    scrs.run_moity_permutations(spec, 0, lock)
    # Best model path recorded
    model_path = scrs.best_model_path or "(persistence only, single run)"
    return {
        "tmpdir": os.path.abspath(scrs.project_name),
        "save_path": model_path,
        "graph_path": None,
        "nodes": None,
        "max_abs_diff": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Cerebros smoke test")
    parser.add_argument("--mode", choices=["keras", "cerebros"], default="keras")
    parser.add_argument("--levels", type=int, default=3, help="Levels (cerebros mode)")
    parser.add_argument("--units", type=int, default=4, help="Units per level (cerebros mode)")
    parser.add_argument("--neurons-min", type=int, default=32)
    parser.add_argument("--neurons-max", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    start = time.time()
    seed = int(os.environ.get("CEREBROS_SMOKE_SEED", 1234))
    set_global_seed(seed, deterministic=False)

    # Extra safeguard: hide GPUs at runtime too (in case already visible)
    try:  # pragma: no cover - environment dependent
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    if args.mode == "keras":
        result = run_basic_keras_smoke(seed)
        elapsed = time.time() - start
        print("=== Cerebros Smoke Success (Keras Mode) ===")
        print(f"Temp dir: {result['tmpdir']}")
        print(f"Seed: {seed}")
        print(f"Saved model: {result['save_path']}")
        print(f"Exported graph nodes: {result['nodes']}")
        print(f"Prediction max abs diff (orig vs reload): {result['max_abs_diff']:.2e}")
        print(f"Elapsed: {elapsed:.2f}s")
    else:
        result = run_cerebros_smoke(
            seed=seed,
            levels=args.levels,
            units_per_level=args.units,
            neurons_min=args.neurons_min,
            neurons_max=args.neurons_max,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - start
        print("=== Cerebros Smoke Success (Cerebros Mode) ===")
        print(f"Project dir: {result['tmpdir']}")
        print(f"Seed: {seed}")
        print(f"Levels: {args.levels} Units/Level: {args.units} Neurons: {args.neurons_min}-{args.neurons_max}")
        print(f"Epochs: {args.epochs} Batch size: {args.batch_size}")
        print(f"Saved model (best path may update post-run): {result['save_path']}")
        print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
