# Authors: Mukund Sudarshan, Nhi Nguyen

import pandas as pd
import numpy as np
import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

from tensorflow.keras.models import Model, load_model
#from tensorflow.keras import Input
from src.figutils import create_input_data, add_flanking, rna_fold_structs
from src.generate_custom_model import generate_custom_model, fit_new_dataset
import src.quad_model
import json
import tensorflow as tf
#from src.quad_model import SlicingOpLambda
#from tensorflow import keras
from src.load_legacy_model import load_legacy_weights_model
#keras.config.enable_unsafe_deserialization()

## CONSTANTS

# model
MODEL_FNAME = "models/custom_adjacency_regularizer_20210731_124_step3.h5"

## MAIN FUNCTION
def get_vis_data(exon, json_file=None, threshold=0.001, use_new_grouping=False, dataset_name="BRCA2_exon_7", new_dataset=None):
    exon = exon.replace("T", "U")
    orig_model = load_legacy_weights_model(MODEL_FNAME, input_length=90)
    
    MODEL_DATA_FNAME = "data/model_data.json"
    if use_new_grouping:
        MODEL_DATA_FNAME = "data/model_data_18.json"
    with open(MODEL_DATA_FNAME, "r") as f:
        model_data = json.load(f)
    DATASET_DATA_FNAME = "data/datasets_data.json"
    with open(DATASET_DATA_FNAME, "r") as f:
        dataset_data = json.load(f)[dataset_name]
    if new_dataset is not None:
        dataset_data = fit_new_dataset(new_dataset)
    pre_flanking_sequence = dataset_data["pre_flanking_sequence"]
    post_flanking_sequence = dataset_data["post_flanking_sequence"]
    basal_shift = dataset_data["basal_shift"]
    model = generate_custom_model(
        new_input_length=len(exon) + len(pre_flanking_sequence) + len(post_flanking_sequence),
        delta_basal=basal_shift,
        orig_model=orig_model
    )

    # Bias
    link_midpoint = model_data["link_midpoint"] + basal_shift
    print(link_midpoint)
    incl_bias, skip_bias = (np.abs(link_midpoint), 0) if link_midpoint < 0 else (0, np.abs(link_midpoint))

    # Number of filters
    num_seq_filters = model_data["num_seq_filters"]
    num_struct_filters = model_data["num_struct_filters"]
    seq_filter_width = model_data["seq_filter_width"]
    struct_filter_width = model_data["struct_filter_width"]
    
    # Filter groups
    incl_seq_groups = model_data["incl_seq_groups"]
    skip_seq_groups = model_data["skip_seq_groups"]
    incl_struct_groups = model_data["incl_struct_groups"]
    skip_struct_groups = model_data["skip_struct_groups"]

    # Filter boundaries
    seq_logo_boundaries = model_data["seq_logo_boundaries"]
    struct_logo_boundaries = model_data["struct_logo_boundaries"]

    # Sequences
    sequence = add_flanking(exon, 
        pre_flanking_sequence=pre_flanking_sequence, 
        post_flanking_sequence=post_flanking_sequence
    )
    structs, _ = rna_fold_structs([sequence])
    structs = structs[0]
    predicted_psi = model.predict(create_input_data([sequence]), verbose=0).item()

    # Compute activations
    activations_model = Model(inputs=model.inputs, outputs=[
        model.get_layer("activation_2").output,
        model.get_layer("activation_3").output
    ])
    data_incl_acts, data_skip_acts = activations_model.predict(create_input_data([sequence]), verbose=0)
    incl_acts = data_incl_acts[0]
    skip_acts = data_skip_acts[0]
    incl_strength = incl_bias + incl_acts.sum()
    skip_strength = skip_bias + skip_acts.sum()
    delta_force = incl_strength - skip_strength

    # Collapse filters
    (
        collapsed_seq_incl_acts, collapsed_struct_incl_acts, 
        collapsed_seq_skip_acts, collapsed_struct_skip_acts
    ) = collapse_activations(
        incl_acts=incl_acts,
        skip_acts=skip_acts,
        incl_seq_groups=incl_seq_groups,
        skip_seq_groups=skip_seq_groups,
        incl_struct_groups=incl_struct_groups,
        skip_struct_groups=skip_struct_groups,
        seq_logo_boundaries=seq_logo_boundaries,
        struct_logo_boundaries=struct_logo_boundaries,
        num_seq_filters=num_seq_filters,
        num_struct_filters=num_struct_filters,
        sequence_length=len(sequence),
    )
    
    # Get hierarchical data
    feature_activations = get_feature_activations(
        collapsed_seq_incl_acts=collapsed_seq_incl_acts,
        collapsed_struct_incl_acts=collapsed_struct_incl_acts, 
        collapsed_seq_skip_acts=collapsed_seq_skip_acts, 
        collapsed_struct_skip_acts=collapsed_struct_skip_acts,
        sequence_length=len(sequence),
        incl_bias=incl_bias,
        skip_bias=skip_bias,
        threshold=threshold
    )

    nucleotide_activations = get_nucleotide_activations(
        collapsed_seq_incl_acts=collapsed_seq_incl_acts,
        collapsed_struct_incl_acts=collapsed_struct_incl_acts, 
        collapsed_seq_skip_acts=collapsed_seq_skip_acts, 
        collapsed_struct_skip_acts=collapsed_struct_skip_acts,
        sequence_length=len(sequence),
        seq_filter_width=seq_filter_width,
        struct_filter_width=struct_filter_width,
        threshold=threshold
    )

    result = {
        "use_new_grouping": int(use_new_grouping),
        "exon": exon,
        "sequence": sequence,
        "structs": structs,
        "predicted_psi": predicted_psi,
        "delta_force": delta_force,
        "incl_bias": incl_bias,
        "skip_bias": skip_bias,
        "incl_strength": incl_strength,
        "skip_strength": skip_strength,
        "feature_activations": {
            "name": "feature_activations",
            "children": feature_activations
        },
        "nucleotide_activations": {
            "name": "nucleotide_activations",
            "children": nucleotide_activations
        },
    }

    if json_file is not None:
        with open(json_file, "w") as f:
            json.dump(result, f, indent=2)
    return result

def shift_row(row, shift, total_len=90):
    shift = int(shift)
    out = np.zeros(total_len)
    out[shift:len(row)+shift] += row
    return out

def collapse(groups, shifted_acts, logo_boundaries, sequence_length):
    collapsed_acts = np.zeros((sequence_length, len(groups), 2))
    for i in range(sequence_length):
        for feature_id in groups.keys():
            filter_strengths = shifted_acts[i,groups[feature_id]]
            feature_strength = filter_strengths.sum()
            repr_filter = groups[feature_id][np.argmax(filter_strengths)]
            feature_length = logo_boundaries[repr_filter]["length"]
            collapsed_acts[i,int(feature_id)-1,:] = [feature_strength, feature_length]
    return collapsed_acts

def collapse_activations(incl_acts, skip_acts, incl_seq_groups, skip_seq_groups,
    incl_struct_groups, skip_struct_groups, seq_logo_boundaries, struct_logo_boundaries,
    num_seq_filters, num_struct_filters, sequence_length
):
    # Shift by start boundaries
    seq_incl_shifts = [seq_logo_boundaries["incl"][i]["left"] for i in range(num_seq_filters)]
    seq_skip_shifts = [seq_logo_boundaries["skip"][i]["left"] for i in range(num_seq_filters)]
    
    struct_incl_shifts = [struct_logo_boundaries["incl"][i]["left"] for i in range(num_struct_filters)]
    struct_skip_shifts = [struct_logo_boundaries["skip"][i]["left"] for i in range(num_struct_filters)]

    # Separate activations
    seq_incl_acts = incl_acts[:,:num_seq_filters]
    seq_skip_acts = skip_acts[:,:num_seq_filters]

    struct_incl_acts = incl_acts[:,num_seq_filters:]
    struct_skip_acts = skip_acts[:,num_seq_filters:]

    # Shift activations
    shifted_seq_incl_acts = np.array([
        shift_row(row, row_shift, sequence_length) 
        for row, row_shift in zip(seq_incl_acts.T, seq_incl_shifts)
    ]).T
    shifted_seq_skip_acts = np.array([
        shift_row(row, row_shift, sequence_length) 
        for row, row_shift in zip(seq_skip_acts.T, seq_skip_shifts)
    ]).T

    left_padding_struct_incl_acts = np.copy(struct_incl_acts[:12,:])
    right_padding_struct_incl_acts = np.copy(struct_incl_acts[-12:,:])
    struct_incl_acts[:12,:] = 0.
    struct_incl_acts[-12:,:] = 0.
    shifted_struct_incl_acts = np.array([
        shift_row(row, row_shift, sequence_length) 
        for row, row_shift in zip(struct_incl_acts.T, struct_incl_shifts)
    ]).T
    shifted_struct_incl_acts[:12,:] += left_padding_struct_incl_acts
    shifted_struct_incl_acts[-12:,:] += right_padding_struct_incl_acts

    left_padding_struct_skip_acts = np.copy(struct_skip_acts[:12,:])
    right_padding_struct_skip_acts = np.copy(struct_skip_acts[-12:,:])
    struct_skip_acts[:12,:] = 0.
    struct_skip_acts[-12:,:] = 0.
    shifted_struct_skip_acts = np.array([
        shift_row(row, row_shift, sequence_length) 
        for row, row_shift in zip(struct_skip_acts.T, struct_skip_shifts)
    ]).T
    shifted_struct_skip_acts[:12,:] += left_padding_struct_skip_acts
    shifted_struct_skip_acts[-12:,:] += right_padding_struct_skip_acts

    # Combine strength; Save the length of strongest filter
    collapsed_seq_incl_acts = collapse(
        groups=incl_seq_groups,
        shifted_acts=shifted_seq_incl_acts,
        logo_boundaries=seq_logo_boundaries["incl"],
        sequence_length=sequence_length
    )
    collapsed_struct_incl_acts = collapse(
        groups=incl_struct_groups,
        shifted_acts=shifted_struct_incl_acts,
        logo_boundaries=struct_logo_boundaries["incl"],
        sequence_length=sequence_length
    )
    collapsed_seq_skip_acts = collapse(
        groups=skip_seq_groups,
        shifted_acts=shifted_seq_skip_acts,
        logo_boundaries=seq_logo_boundaries["skip"],
        sequence_length=sequence_length
    )
    collapsed_struct_skip_acts = collapse(
        groups=skip_struct_groups,
        shifted_acts=shifted_struct_skip_acts,
        logo_boundaries=struct_logo_boundaries["skip"],
        sequence_length=sequence_length
    )

    return (
        collapsed_seq_incl_acts, collapsed_struct_incl_acts, 
        collapsed_seq_skip_acts, collapsed_struct_skip_acts
    )

def transform(d, parent):
    return (
        {"name": parent, "strength": d[parent]["strength"], "length": d[parent]["length"]}
            if "strength" in d[parent].keys() else
        {"name": parent, "children": [transform(d[parent], child) for child in d[parent]]}
    )

def get_feature_activations_helper(collapsed_acts, acts_dict, name, sequence_length, threshold):
    for fi in range(collapsed_acts.shape[1]):
        acts_dict[f"{name}_{fi+1}"] = {}
        for i in range(sequence_length):
            feature_strength = collapsed_acts[i,fi,0]
            feature_length = int(collapsed_acts[i,fi,1])
            pos_ind = i+1
            if "struct" in name: 
                # Ignore the first 12 and the last 12 struct act
                if i < 12 or i > 77:
                    feature_length = 1
                else:
                    pos_ind -= 12
            if feature_strength > threshold:
                acts_dict[f"{name}_{fi+1}"][f"pos_{pos_ind}"] = {
                    "strength": feature_strength,
                    "length": feature_length,
                }
                
def get_feature_activations(collapsed_seq_incl_acts, collapsed_struct_incl_acts, 
    collapsed_seq_skip_acts, collapsed_struct_skip_acts, sequence_length, incl_bias, skip_bias, 
    threshold=0.001
):
    feature_activations = {
        "incl": {"incl_bias": {"strength": incl_bias, "length": 0}},
        "skip": {"skip_bias": {"strength": skip_bias, "length": 0}}
    }
    get_feature_activations_helper(collapsed_seq_incl_acts, feature_activations["incl"], 
        "incl", sequence_length, threshold)
    get_feature_activations_helper(collapsed_struct_incl_acts, feature_activations["incl"], 
        "incl_struct", sequence_length, threshold)
    get_feature_activations_helper(collapsed_seq_skip_acts, feature_activations["skip"], 
        "skip", sequence_length, threshold)
    get_feature_activations_helper(collapsed_struct_skip_acts, feature_activations["skip"], 
        "skip_struct", sequence_length, threshold)
    return [
        transform(feature_activations, "incl"),
        transform(feature_activations, "skip"),
    ]

def get_nucleotide_activations_helper(collapsed_acts, acts_dict, name, sequence_length, 
    filter_width, threshold,    
):
    for i in range(sequence_length):
        for fi in range(collapsed_acts.shape[1]):
            fix_feature_length = sequence_length
            pos_ind = i + 1
            if "struct" in name:
                # Ignore the first 12 and the last 12 struct act
                if i < 12 or i > 77:
                    first_start_ind = i
                    fix_feature_length = 1
                else:
                    first_start_ind = max(12, i-filter_width+1)
            else:
                first_start_ind = max(0, i-filter_width+1)
            for start_ind in range(first_start_ind,i+1):
                feature_strength = collapsed_acts[start_ind,fi,0]
                feature_length = min(fix_feature_length, int(collapsed_acts[start_ind,fi,1]))
                if feature_strength > threshold and start_ind + feature_length > i:
                    if f"{name}_{fi+1}" not in acts_dict[f"pos_{pos_ind}"].keys():
                        acts_dict[f"pos_{pos_ind}"][f"{name}_{fi+1}"] = {}
                    acts_dict[f"pos_{pos_ind}"][f"{name}_{fi+1}"][f"feature_pos_{i-start_ind+1}"] = {
                        "strength": feature_strength/feature_length,
                        "length": feature_length,
                    }

def get_nucleotide_activations(collapsed_seq_incl_acts, collapsed_struct_incl_acts, 
    collapsed_seq_skip_acts, collapsed_struct_skip_acts, sequence_length, 
    seq_filter_width, struct_filter_width, threshold=0.001
):
    nucleotide_activations = {"incl": {}, "skip": {}}
    for i in range(sequence_length):
        nucleotide_activations["incl"][f"pos_{i+1}"] = {}
        nucleotide_activations["skip"][f"pos_{i+1}"] = {}
    get_nucleotide_activations_helper(collapsed_seq_incl_acts, nucleotide_activations["incl"], 
        "incl", sequence_length, seq_filter_width, threshold)
    get_nucleotide_activations_helper(collapsed_struct_incl_acts, nucleotide_activations["incl"], 
        "incl_struct", sequence_length, struct_filter_width, threshold)
    get_nucleotide_activations_helper(collapsed_seq_skip_acts, nucleotide_activations["skip"], 
        "skip", sequence_length, seq_filter_width, threshold)
    get_nucleotide_activations_helper(collapsed_struct_skip_acts, nucleotide_activations["skip"], 
        "skip_struct", sequence_length, struct_filter_width, threshold)
    for i in range(sequence_length):
        if len(nucleotide_activations["incl"][f"pos_{i+1}"]) == 0:
            nucleotide_activations["incl"].pop(f"pos_{i+1}")
        if len(nucleotide_activations["skip"][f"pos_{i+1}"]) == 0:
            nucleotide_activations["skip"].pop(f"pos_{i+1}")
    return [
        transform(nucleotide_activations, "incl"),
        transform(nucleotide_activations, "skip"),
    ]