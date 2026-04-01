# Author: Mukund Sudarshan, Nhi Nguyen

## The generate_custom_model function will generate a model with a custom exon length by Lanczos resampling of the position bias coefficients. It also allows to modify the bias term in the SumDiff layer.

import numpy as np


import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

from src.quad_model import *
from tensorflow.keras.models import Model, load_model
from scipy.stats import linregress
from src.figutils import create_input_data
from src.quad_model import get_model
import tensorflow as tf

ORIG_MODEL_FILE_NAME = 'models/custom_adjacency_regularizer_20210731_124_step3.h5'

def lanczos_kernel(x, order):
    return np.sinc(x) * np.sinc(x/order) * ((x > -order) * (x < order))

# arr is the original array (as numpy array)
# positions are the positions to interpolate through (with integer positions corresponding to original indices of samples)
# "order" is typically 2 or 3
def lanczos_interpolate(arr, positions, order=3):
    result = np.zeros_like(positions)
    # Written this way to support non-scalar x.
    for i, x in enumerate(positions):
        i_min, i_max = int(np.floor(x) - order +1), int(np.floor(x) + order + 1)
        i_min, i_max = max(i_min, 0), min(i_max, len(arr))
        window = np.arange(i_min, i_max)
        result[i] = np.sum(arr[window] * lanczos_kernel(x - window, order))

    return result

def lanczos_resampling(arr, new_len, order=3):
    return lanczos_interpolate(arr, np.linspace(0, len(arr)-1, num=new_len), order)

# Given a 1D array orig_weights, resample its central region to the new weight; the first and last padding elements are kept as is
def resample_one_positional_bias(orig_weights, new_input_length, padding):
    assert(new_input_length > 2*padding)
    return np.concatenate((orig_weights[:padding],
            lanczos_resampling(orig_weights[padding:-padding], new_input_length - 2*padding),
            orig_weights[-padding:]))
    

def resample_positional_bias_weights(orig_weights, new_input_length, padding):
    return np.apply_along_axis(lambda x: resample_one_positional_bias(x, new_input_length, padding), 0, orig_weights)

# Generate a new model accepting a new input length (originally it was 90) and adding a delta_basal to the final activation before the tuner
def generate_custom_model(new_input_length, delta_basal,orig_model):
    tf.keras.backend.clear_session()
    new_model = get_model(input_length=new_input_length)

    orig_model_layer_names = [layer.name for layer in orig_model.layers]
    new_names = {l.name for l in new_model.layers}
    orig_names = {l.name for l in orig_model.layers}

    print("In new not in orig:", sorted(new_names - orig_names)[:30])
    print("In orig not in new:", sorted(orig_names - new_names)[:30])
    for layer in new_model.layers:
        new_w = layer.get_weights()
        if not new_w:
            continue

        if layer.name not in orig_model_layer_names:
            print(f"[WARN] No matching layer in orig_model: {layer.name} ({layer.__class__.__name__})")
            continue

        orig_layer = orig_model.get_layer(layer.name)
        orig_w = orig_layer.get_weights()

        if "position" not in layer.name:
            layer.set_weights(orig_w)
        else:
            layer.set_weights([resample_positional_bias_weights(
                orig_w[0],
                new_w[0].shape[0],
                15
            )])

    w, b = new_model.get_layer("energy_seq_struct").get_weights()
    new_model.get_layer("energy_seq_struct").set_weights([w + delta_basal, b])
    return new_model

# Fit a custom model to a new dataset
def fit_new_dataset(new_dataset):
    input_to_model = create_input_data(list(new_dataset.sequence))
    seq_len = len(new_dataset.sequence.iloc[0])
    best_rmse = 1000
    r2 = None
    for basal_shift in np.arange(-10,10,0.2):
        basal_shift = np.round(basal_shift,1)
        custom_model = generate_custom_model(seq_len, basal_shift)
        predictions = custom_model(input_to_model).numpy().flatten()
        rmse = ((new_dataset.PSI-predictions)**2).mean()
        if (rmse<best_rmse):
            best_basal_shift = basal_shift
            best_rmse = rmse
            r2 = r2_score(new_dataset.PSI, predictions)
    sequence = new_dataset.sequence.iloc[0]
    exon = new_dataset.exon.iloc[0]
    pre_flanking_sequence = sequence[:sequence.find(exon)]
    post_flanking_sequence = sequence[(sequence.find(exon)+len(exon)):]
    return {
        "basal_shift": best_basal_shift,
        "pre_flanking_sequence": pre_flanking_sequence,
        "post_flanking_sequence": post_flanking_sequence,
        "number_of_datapoints": len(new_dataset),
        "exon_length": len(new_dataset.exon.iloc[0]),
        "sequence_length": seq_len,
        "r_squared": r2
    }

def r2_score(y_true, y_pred):
    slope, intercept, r_value, p_value, std_err = linregress(y_pred, y_true)
    return r_value