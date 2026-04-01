# Authors: Mukund Sudarshan, Nhi Nguyen 

import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Input
from src.figutils import create_input_data, add_flanking
import src.quad_model
import json

## CONSTANTS

# collapse filters
INCL_SEQ_GROUPS = {
    1: [6, 9, 11, 14],
    3: [4, 5, 8, 16, 17],
    7: [2, 12, 13],
    5: [3, 10, 15],
    6: [1, 7],
    2: [0, 19],
    4: [18],
}
SKIP_SEQ_GROUPS = {
    6: [1, 5, 9, 19], 
    4: [3, 10, 14, 15, 18],  
    3: [2, 4, 8, 13],     
    5: [6, 7, 12],     
    1: [0, 17],                    
    2: [11, 16],                   
}
INCL_STRUCT_GROUPS = {
    1: np.array([0, 1, 2, 3, 4, 5, 6, 7]), 
}
SKIP_STRUCT_GROUPS = {
    1: np.array([1]), 
    2: np.array([0, 2, 3]), 
    3: np.array([5, 6, 7]), 
    4: np.array([4])
}

# shift filters
INCL_SHIFTS = [
    2, 1, 0, 3, 2, 1, 0, 1, 0, 1, 2, 1, 0, 2, 0, 2, 0, 2, 1, 1, # inclusion sequence
    0, 0, 0, 0, 0, 0, 0, 0  # inclusion structure
]
SKIP_SHITS = [
    1, 0, 2, 2, 3, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 2, 0, 3, 0, 1, # skipping sequence
    0, 0, 0, 0, 0, 0, 0, 0  # skipping structure
]

# filter names
FILTER_NAMES = {
    "incl_seq_6_9_11_14"            : "incl_1",
    "incl_seq_4_5_8_16_17"          : "incl_2",
    "incl_seq_2_12_13"              : "incl_3",
    "incl_seq_3_10_15"              : "incl_4",
    "incl_seq_1_7"                  : "incl_5",
    "incl_seq_0_19"                 : "incl_6",
    "incl_seq_18"                   : "incl_7",         # not in the paper
    "incl_struct_0_1_2_3_4_5_6_7"   : "incl_1_struct",  # not in the paper
    "skip_seq_1_5_9_19"             : "skip_7",
    "skip_seq_3_10_14_15_18"        : "skip_8",
    "skip_seq_2_4_8_13"             : "skip_9",
    "skip_seq_0_17"                 : "skip_10",
    "skip_seq_6_7_12"               : "skip_11",
    "skip_seq_11_16"                : "skip_12",        # not in the paper
    "skip_struct_0_2_3"             : "skip_s",         
    "skip_struct_1"                 : "skip_p",
    "skip_struct_4"                 : "skip_blank",     # not in the paper
    "skip_struct_5_6_7"             : "skip_dot",       # not in the paper
}

# model
MODEL_FNAME = "../models/custom_adjacency_regularizer_20210731_124_step3.h5"

## MAIN FUNCTION
def get_deciphering_rna_splicing_data(exons, json_file=None):
    N = len(exons)
    
    # Model
    custom_model = load_model(MODEL_FNAME)
    num_seq_filters = custom_model.get_layer("qc_incl").kernel.shape[2]
    link_midpoint = get_model_midpoint(custom_model)

    # Sequences
    sequences = [add_flanking(e, 10) for e in exons]

    # Compute activations for given sequences
    activations_model = Model(inputs=custom_model.inputs, outputs=[
        custom_model.get_layer("activation_2").output,
        custom_model.get_layer("activation_3").output
    ])
    data_incl_act, data_skip_act = activations_model.predict(create_input_data(sequences))

    # Compute forces
    json_data = []
    for idx in range(N):
        exon = exons[idx]
        sequence = sequences[idx]
        incl_act = data_incl_act[idx]
        skip_act = data_skip_act[idx]

        shifted_incl_acts = np.array([
            shift_row(row, row_shift) for row, row_shift in zip(incl_act.T, INCL_SHIFTS)
        ]).T
        shifted_skip_acts = np.array([
            shift_row(row, row_shift) for row, row_shift in zip(skip_act.T, SKIP_SHITS)
        ]).T

        iT, sT = collapse_filters(
            shifted_incl_acts, 
            shifted_skip_acts, 
            iM=INCL_SEQ_GROUPS, 
            sM=SKIP_SEQ_GROUPS, 
            iM_struct=INCL_STRUCT_GROUPS, 
            sM_struct=SKIP_STRUCT_GROUPS, 
            num_seq_filters=num_seq_filters
        )
        if link_midpoint < 0:
            incl_bias = np.abs(link_midpoint)
            skip_bias = 0
        else:
            skip_bias = np.abs(link_midpoint)
            incl_bias = 0
        
        delta_force = incl_bias - skip_bias + iT.sum(axis=0).sum() - sT.sum(axis=0).sum()
        predicted_psi = custom_model.predict(create_input_data([sequences[idx]])).item()


        exon_data = {
            "exon_id": idx,
            "exon": exon,
            "sequence": sequence,
            "predicted_psi": predicted_psi,
            "delta_force": delta_force,
            "incl_bias": incl_bias,
            "skip_bias": skip_bias,
            "activations": []
        }

        # Filter names
        filter_names = (
            [FILTER_NAMES[iT_column] for iT_column in iT.columns] 
            + [FILTER_NAMES[sT_column] for sT_column in sT.columns]
        )
        
        for i in range(iT.shape[0]):
            position = {"position": i}
            strengths = list(iT.iloc[i,:]) + list(sT.iloc[i,:])
            activations = {filter_names[i]: strengths[i] for i in range(len(filter_names))}
            exon_data["activations"].append({**position, **activations})

        json_data.append(exon_data)

    if json_file is not None:
        with open(json_file, "w") as f:
            json.dump(json_data, f)
    return json_data


## HELPERS
def get_link_midpoint(link_function, midpoint=0.5, epsilon=1e-5, lb=-100, ub=100, max_iters=50):
    """
    Assumes monotonicity and smoothness of link function
    """
    iters = 0
    while iters < max_iters:
        xx = np.linspace(lb, ub, 1000)
        yy = link_function(xx[:, None]).numpy().flatten()

        if min(np.abs(yy - midpoint)) < epsilon:
            return xx[np.abs(yy - midpoint) < epsilon][0]
        lb_idx = np.where((yy - midpoint) < 0)[0][-1]
        ub_idx = np.where((yy - midpoint) > 0)[0][0]

        lb = xx[lb_idx]
        ub = xx[ub_idx]

        iters += 1
    raise RuntimeError(f"Max iterations ({max_iters}) reached without solution...")

def get_model_midpoint(model, midpoint=0.5):    
    """ 
    Compute the midpoint using the model"s link function. This is the negation of the basal strength. 
    I.e., positive value corresponds to a skipping basal strength. 
    """
    link_input = Input(shape=(1,))
    w = model.get_layer("energy_seq_struct").w.numpy()
    b = model.get_layer("energy_seq_struct").b.numpy()
    link_output = model.get_layer("output_activation")(model.get_layer("gen_func")(w*link_input + b))
    link_function = Model(inputs=link_input, outputs=link_output)
    return get_link_midpoint(link_function, midpoint)

def shift_row(row, shift, total_len=90):
    out = np.zeros(total_len)
    offset = (total_len - len(row)) // 2
    out[offset+shift:offset+len(row)+shift] += row
    return out

def collapse_filters(test_act_incl, test_act_skip, iM, sM, iM_struct, sM_struct, num_seq_filters):
    """
    Collapse filters with the same motif.
    Collapse 20 inclusion filters to 8; Collapse 20 skipping filters to 10.
    """

    incl_seq_filter_names = [
        "incl_seq_" + "_".join(map(str, iM[key])) for key in iM.keys()
    ]
    incl_struct_filter_names = [
        "incl_struct_" + "_".join(map(str, iM_struct[key])) for key in iM_struct.keys()
    ]
    skip_seq_filter_names = [
        "skip_seq_" + "_".join(map(str, sM[key])) for key in sM.keys()
    ]
    skip_struct_filter_names = [
        "skip_struct_" + "_".join(map(str, sM_struct[key])) for key in sM_struct.keys()
    ]

    nf_i = len(incl_seq_filter_names) + len(incl_struct_filter_names)
    nf_s = len(skip_seq_filter_names) + len(skip_struct_filter_names)

    test_act_incl_collapsed = np.zeros((test_act_incl.shape[0], nf_i))
    test_act_skip_collapsed = np.zeros((test_act_skip.shape[0], nf_s))

    ctr = 0
    for incl_idx, incl_key in enumerate(iM.keys()):
        test_act_incl_collapsed[:, incl_idx] = test_act_incl[:, iM[incl_key]].sum(
            axis=1
        )
        ctr += 1

    for incl_idx, incl_key in enumerate(iM_struct.keys()):
        test_act_incl_collapsed[:, ctr + incl_idx] = test_act_incl[
            :, num_seq_filters + iM_struct[incl_key]
        ].sum(axis=1)

    ctr = 0
    for skip_idx, skip_key in enumerate(sM.keys()):
        test_act_skip_collapsed[:, skip_idx] = test_act_skip[:, sM[skip_key]].sum(
            axis=1
        )
        ctr += 1

    for skip_idx, skip_key in enumerate(sM_struct.keys()):
        test_act_skip_collapsed[:, ctr + skip_idx] = test_act_skip[
            :, num_seq_filters + sM_struct[skip_key]
        ].sum(axis=1)

    return pd.DataFrame(
        test_act_incl_collapsed,
        columns=incl_seq_filter_names + incl_struct_filter_names,
    ), pd.DataFrame(
        test_act_skip_collapsed,
        columns=skip_seq_filter_names + skip_struct_filter_names,
    )