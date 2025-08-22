
import numpy as np
import argparse
import sys
# from multiprocessing import Pool  # , Process
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
import pendulum
import pandas as pd
import tensorflow as tf
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval

NUMBER_OF_TRAILS_PER_BATCH = 2
NUMBER_OF_BATCHES_OF_TRIALS = 2

###

LABEL_COLUMN = 'price'

## your data:


TIME = pendulum.now().__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test'

# ---------------- CLI ARGS (chart network graph) ----------------
parser = argparse.ArgumentParser(description="Ames housing Cerebros random search (no preprocessing)")
parser.add_argument("--chart-network-graph", "--shart-network-graph", dest="chart_network_graph", action="store_true", help="Enable plotting/rendering of the neural network graph (accepts legacy typo variant).")
parser.add_argument("--no-chart-network-graph", dest="chart_network_graph", action="store_false", help="Disable plotting of the neural network graph.")
parser.set_defaults(chart_network_graph=False)
known_args, _unknown = parser.parse_known_args()

# Accept key=value forms: chart_network_graph=true / shart_network_graph=true
def _coerce_bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y", "on")
for raw in sys.argv[1:]:
    if "=" not in raw:
        # Accept bare tokens like 'chart-network-graph' / 'shart-network-graph'
        token = raw.strip().lower()
        if token in ("chart-network-graph", "shart-network-graph"):
            known_args.chart_network_graph = True
        continue
    k, v = raw.split("=", 1)
    nk = k.strip().lower().replace("-", "_")
    if nk in ("chart_network_graph", "shart_network_graph") and _coerce_bool(v):
        known_args.chart_network_graph = True


# white = pd.read_csv('wine_data.csv')

raw_data = pd.read_csv('ames.csv')
needed_cols = [
    col for col in raw_data.columns 
    if raw_data[col].dtype != 'object' 
    and col != LABEL_COLUMN]
data_numeric = raw_data[needed_cols].fillna(0).astype(float)
label = raw_data.pop(LABEL_COLUMN)

data_np = data_numeric.values

tensor_x =\
    tf.constant(data_np)

training_x = [tensor_x]

INPUT_SHAPES = [training_x[i].shape[1] for i in np.arange(len(training_x))]

train_labels = [tf.constant(label.values.astype(float))]

OUTPUT_SHAPES = [1]  # [train_labels[i].shape[1]


# Params for a training function (Approximately the oprma
# discovered in a bayesian tuning study done on Katib)

meta_trial_number = 0  # In distributed training set this to a random number
activation = 'swish'
predecessor_level_connection_affinity_factor_first = 0.506486683067576
predecessor_level_connection_affinity_factor_main = 1.6466748663373876
max_consecutive_lateral_connections = 35
p_lateral_connection = 3.703218275217572
num_lateral_connection_tries_per_unit = 12
learning_rate = 0.02804912925494706
epochs = 130
batch_size = 78
maximum_levels = 4
maximum_units_per_level = 3
maximum_neurons_per_unit = 3


cerebros =\
    SimpleCerebrosRandomSearch(
        unit_type=DenseUnit,
        input_shapes=INPUT_SHAPES,
        output_shapes=OUTPUT_SHAPES,
        training_data=training_x,
        labels=train_labels,
        validation_split=0.35,
        direction='minimize',
        metric_to_rank_by='val_root_mean_squared_error',
        minimum_levels=4,
        maximum_levels=maximum_levels,
        minimum_units_per_level=2,
        maximum_units_per_level=maximum_units_per_level,
        minimum_neurons_per_unit=3,
        maximum_neurons_per_unit=maximum_neurons_per_unit,
        activation=activation,
        final_activation=None,
        number_of_architecture_moities_to_try=7,
        number_of_tries_per_architecture_moity=1,
        number_of_generations=3,
        minimum_skip_connection_depth=1,
        maximum_skip_connection_depth=7,
        predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
        predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
        predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
        predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
        predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
        seed=8675309,
        max_consecutive_lateral_connections=max_consecutive_lateral_connections,
        gate_after_n_lateral_connections=3,
        gate_activation_function=simple_sigmoid,
        p_lateral_connection=p_lateral_connection,
        p_lateral_connection_decay=zero_95_exp_decay,
        num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
        learning_rate=learning_rate,
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
        epochs=epochs,
        patience=7,
        project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
        # use_multiprocessing_for_multiple_neural_networks=False,  # pull this param
        model_graphs='model_graphs',
        batch_size=batch_size,
        meta_trial_number=meta_trial_number,
        chart_network_graph=known_args.chart_network_graph)
result = cerebros.run_random_search()

print("Best model: (May need to re-initialize weights, and retrain with early stopping callback)")
best_model_found = cerebros.get_best_model()
print(best_model_found.summary())

print("result extracted from cerebros")
print(f"Final result was (val_root_mean_squared_error): {result}")
