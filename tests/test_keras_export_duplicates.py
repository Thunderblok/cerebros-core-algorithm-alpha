import os
# Force CPU for deterministic behavior
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from cerebros.keras_export import export_keras_to_graph

# Also disable GPUs at the TensorFlow level in case the env var is ignored
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def test_concatenate_preserves_duplicate_inputs():
    inp = tf.keras.Input(shape=(4,), name="inp")
    a = tf.keras.layers.Dense(2, activation="linear", name="a")(inp)
    b = tf.keras.layers.Dense(5, activation="linear", name="b")(inp)

    cat = tf.keras.layers.Concatenate(axis=1, name="cat")([inp, inp, a, a, b, b])
    out = tf.keras.layers.Dense(1, name="out")(cat)
    model = tf.keras.Model(inputs=inp, outputs=out)

    spec = export_keras_to_graph(model)
    # Find the cat node
    cat_nodes = [n for n in spec["nodes"] if n["name"] == "cat"]
    assert len(cat_nodes) == 1
    inputs = cat_nodes[0]["inputs"]
    assert inputs == ["inp", "inp", "a", "a", "b", "b"], inputs
