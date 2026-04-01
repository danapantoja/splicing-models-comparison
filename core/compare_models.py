## 5 models: SpliceAI (uses 5 models), AlphaGenome, MaxEntScan, Interpretable Splcing Model, HAL

from __future__ import annotations
import os, sys

from typing import Any, Dict

import math
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.vis_data import get_vis_data
from alphagenome.data import genome
from alphagenome.models import dna_client

from maxentpy import maxent
from tensorflow.keras.models import load_model
import traceback

# helper functions
def clamp_psi(x: float) -> float:
    """Clamp score to [0, 1] if needed."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return max(0.0, min(1.0, float(x)))


def safe_abs_error(pred: float, actual: float) -> float:
    if pd.isna(pred) or pd.isna(actual):
        return float("nan")
    return abs(pred - actual)


def safe_signed_error(pred: float, actual: float) -> float:
    if pd.isna(pred) or pd.isna(actual):
        return float("nan")
    return pred - actual

## CONFIGURATION

# ES7 construct sequence (from generate_es9_sequences.py)
ES7_SEQUENCE = "ccacctgacgtcgacggatcgggagatcccagtgtggtggtacttacttgtacagctcgtccatgccgagagtgatcccggcggcggtcacgaactccagcaggaccatgtgatcgcgcttctcgttggggtctttgctcagggcggactgggtgctcaggtagtggttgtcgggcagcagcacggggccgtcgccgatgggggtgttctgctggtagtggtcggcgagctgcacgctgccgtcctcgatgttgtggcggatcttgaagttcgccttgatgccgttcttctgcttgtcggccatgatatagacgttgtggctgttgtagttgtactccagcttgtgccccaggatgttgccgtcctccttgaagtcgatgcccttcagctcgatgcggttcaccagggtgtcgccctcgaacttcacctcggcgcgggtcttgtagttgccgtcgtccttgaagaagatggtgcgctcctggacgtagccttcgggcatggcggacttgaagaagtcgtgctgcttcatgtggtcggggtagcggctgaagcactgcacgccgtaggtcagggtggtcacgagggtgggccagggcacgggcagcttgccggtggtgcagatgaacttcagggtcagcttgccgtaggtggcatcgccctcgccctcgccagacacgctgaacttgtggccgtttacgtcgccgtccagctcgaccaggatgggcaccaccccggtgaacagctcctcgcccttgctcaccatggtggcctcacgacacctgaaatggaagaaaaaaactttgaaccactgtctgaggcttgagaatgaaccaagatccaaactcaaaaagggcaaattccaaggagaattacatcaagtgccaagctggcctaacttcagtctccacccactcagtgtggggaaactccatcgcataaaacccctccccccaacctaaagacgacgtactccaaaagctcgagaactaatcgaggtgcctggacggcgcccggtactccgtggagtcacatgaagcgacggctgaggacggaaaggcccttttcctttgtgtgggtgactcacccgcccgctctcccgagcgccgcgtcctccattttgagctccctgcagcagggccgggaagcggccatctttccgctcacgcaactggtgccgaccgggccagccttgccgcccagggcggggcgatacacggcggcgcgaggccaggcaccagagcaggccggccagcttgagactacccccgtccgattctcggtggccgcgctcgcaggccccgcctcgccgaacatgtgcgctgggacgcacgggccccgtcgccgcccgcggccccaaaaaccgaaataccagtgtgcagatcttggcccgcatttacaagactatcttgccagaaaaaaagcgtcgcagcaggtcatcaaaaattttaaatggctagagacttatcgaaagcagcgagacaggcgcgaaggtgccaccagattcgcacgcggcggccccagcgcccaggccaggcctcaactcaagcacgaggcgaaggggctccttaagcgcaaggcctcgaactctcccacccacttccaacccgaagctcgggatcaagaatcacgtactgcagccaggtggaagtaattcaaggcacgcaagggccataacccgtaaagaggccaggcccgcgggaaccacacacggcacttacctgtgttctggcggcaaacccgttgcgaaaaagaacgttcacggcgactactgcacttatatacggttctcccccaccctcgggaaaaaggcggagccagtacacgacatcactttcccagtttaccccgcgccaccttctctaggcaccggttcaattgccgacccctccccccaacttctcggggactgtgggcgatgtgcgctctgcccactgacgggcaccggagcctacgtaggaattaattcgccagcacagtggtcgaggtgagccccacgttctgcttcactctccccatctcccccccctccccacccccaattttgtatttatttattttttaattattttgtgcagcgatgggggcggggggggggggggggcgcgcgccaggcggggcggggcggggcgaggggcggggcggggcgaggcggaaaggtgcggcggcagccaatcaaagcggcgcgctccgaaagtttccttttatggcgaggcggcggcggcggcggccctataaaaagcgaagcgcgcggcgggcgggagtcgctgcgcgctgccttcgccccgtgccccgctccgccgccgcctcgcgccgcccgccccggctctgactgaccgcgttactcccacaggtgagcgggcgggacggcccttctcctccgggctgtaattagcgcttggtttaatgacggctcgtttcttttctgtggctgcgtgaaagccttgaggggctccgggagggccctttgtgcggggggagcggctcggggggtgcgtgcgtgtgtgtgtgcgtggggagcgccgcgtgcggctccgcgctgcccggcggctgtgagcgctgcgggcgcggcgcggggctttgtgcgctccgcagtgtgcgcgaggggagcgcggccgggggcggtgccccgcggtgcggggggggctgcgaggggaacaaaggctgcgtgcggggtgtgtgcgtgggggggtgagcagggggtgtgggcgcgtcggtcgggctgcaaccccccctgcacccccctccccgagttgctgagcacggcccggcttcgggtgcggggctccgtacggggcgtggcgcggggctcgccgtgccgggcggggggtggcggcaggtgggggtgccgggcggggcggggccgcctcgggccggggagggctcgggggaggggcgcggcggcccccggagcgccggcggctgtcgaggcgcggcgagccgcagccattgccttttatggtaatcgtgcgagagggcgcagggacttcctttgtcccaaatctgtgcggagccgaaatctgggaggcgccgccgcaccccctctagcgggcgcggggcgaagcggtgcggcgccggcaggaaggaaatgggcggggagggccttcgtgcgtcgccgcgccgccgtccccttctccctctccagcctcggggctgtccgcggggggacggctgccttcgggggggacggggcagggcggggttcggcttctggcgtgtgaccggcggctctagcgtttaaacttaagctaatacgactcactatagggagcttggtaccgcaacctcaaacagacaccatggtgcacctgactcctgaggagaagtctgccgttactgccctgtggggcaaggtgaacgtggatgaagttggtggtgaggccctgggcaggttggtatcaaggttacaagacaggtttaaggagaccaatagaaactgggcatatggagacagagaagactcttgggtttctgataggcactgactctctctgcctatgtctttctctgccatccaggttnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnncaggtctgactatgggacccttgatgttttctttccccttcttttctatggttaagttcatgtcataggaaggggagaagtaacagggtacagtttagaatgggaaacagacgaatgattgcatcagtgtggaagtctcaggatcgttttagtttcttttatttgctgttcataacaattgttttcttttgtttaattcttgctttctttttttttcttctccgcaatttttactattatacttaatgccttaacattgtgtataacaaaaggaaatatctctgagatacattaagtaacttaaaaaaaaactttacacagtctgcctagtacattactatttggaatatatgtgtgcttatttgcatattcataatctccctactttattttcttttatttttaattgatacataatcattatacatatttatgggttaaagtgtaatgttttaatatcgatacacatattgaccaaatcagggtaattttgcatttgtaattttaaaaaatgctttcttcttttaatatacttttttgtttatcttatttctaatactttccctaatctctttctttcagggcaataatgatacaatgtatcatgcctctttgcaccattctaaagaataacagtgataatttctgggttaaggcaatagcaatatttctgcatataaatatttctgcatataaattgtaactgatgtaagaggtttcatattgctaatagcagctacaatccagctaccattctgcttttattttatggttgggataaggctggattattctgagtccaagctaggcccttttgctaatcatgttcatacctcttatcttcctcccacagctcctgggcaacgtgctggtctgtgtgctggcccatcactttggcaaagaattcaccccaccagtgcaggctgcctatcagaaagtggtggctggtgtggctaatgccctggcccacaagtatcactaagctcgctctagannnnnnnnnnnnnnnnnnnnatagggcccgtttaaacccgctgatcagcctcgactgtgccttctagttgccagccatctgttgtttgcccctcccccgtgccttccttgaccctggaaggtgccactcccactgtcctttcctaataaaatgaggaaattgcatcgcattgtctgagtaggtgtcattctattctggggggtggggtggggcaggacagcaagggggaggattgggaagacaatagcaggcatgctggggatgcggtgggctctatggcttctgaggcggaaagaaccagctggggctctagggggtatccccacgcgccctgtagcggcgcattaagcgcggcgggtgtggtggttacgcgcagcgtgaccgctacacttgccagcgccctagcgcccgctcctttcgctttcttcccttcctttctcgccacgttcgccggctttccccgtcaagctctaaatcggggcatccctttagggttccgatttagtgctttacggcacctcgaccccaaaaaacttgattagggtgatggttcacgtagtgggccatcgccctgatagacggtttttcgccctttgacgttggagtccacgttctttaatagtggactcttgttccaaactggaacaacactcaaccctatctcggtctattcttttgatttataagggattttggggatttcggcctattggttaaaaaatgagctgatttaacaaaaatttaacgcgaattaattctgtggaatgtgtgtcagttagggtgtggaaagtccccaggctccccaggcaggcagaagtatgcaaagcatgcatctcaattagtcagcaaccaggtgtggaaagtccccaggctccccagcaggcagaagtatgcaaagcatgcatctcaattagtcagcaaccatagtcccgcccctaactccgcccatcccgcccctaactccgcccagttccgcccattctccgccccatggctgactaattttttttatttatgcagaggccgaggccgcctctgcctctgagctattccagaagtagtgaggaggcttttttggaggcctaggcttttgcaaaaagctcccgggagcttgtatatccattttcggatctgatcagcacgtgttgacaattaatcatcggcatagtatatcggcatagtataatacgacaaggtgaggaactaaaccatggccaagttgaccagtgccgttccggtgctcaccgcgcgcgacgtcgccggagcggtcgagttctggaccgaccggctcgggttctcccgggacttcgtggaggacgacttcgccggtgtggtccgggacgacgtgaccctgttcatcagcgcggtccaggaccaggtggtgccggacaacaccctggcctgggtgtgggtgcgcggcctggacgagctgtacgccgagtggtcggaggtcgtgtccacgaacttccgggacgcctccgggccggccatgaccgagatcggcgagcagccgtgggggcgggagttcgccctgcgcgacccggccggcaactgcgtgcacttcgtggccgaggagcaggactgacacgtgctacgagatttcgattccaccgccgccttctatgaaaggttgggcttcggaatcgttttccgggacgccggctggatgatcctccagcgcggggatctcatgctggagttcttcgcccaccccaacttgtttattgcagcttataatggttacaaataaagcaatagcatcacaaatttcacaaataaagcatttttttcactgcattctagttgtggtttgtccaaactcatcaatgtatcttatcatgtctgtataccgtcgacctctagctagagcttggcgtaatcatggtcatagctgtttcctgtgtgaaattgttatccgctcacaattccacacaacatacgagccggaagcataaagtgtaaagcctggggtgcctaatgagtgagctaactcacattaattgcgttgcgctcactgcccgctttccagtcgggaaacctgtcgtgccagctgcattaatgaatcggccaacgcgcggggagaggcggtttgcgtattgggcgctcttccgcttcctcgctcactgactcgctgcgctcggtcgttcggctgcggcgagcggtatcagctcactcaaaggcggtaatacggttatccacagaatcaggggataacgcaggaaagaacatgtgagcaaaaggccagcaaaaggccaggaaccgtaaaaaggccgcgttgctggcgtttttccataggctccgcccccctgacgagcatcacaaaaatcgacgctcaagtcagaggtggcgaaacccgacaggactataaagataccaggcgtttccccctggaagctccctcgtgcgctctcctgttccgaccctgccgcttaccggatacctgtccgcctttctcccttcgggaagcgtggcgctttctcaatgctcacgctgtaggtatctcagttcggtgtaggtcgttcgctccaagctgggctgtgtgcacgaaccccccgttcagcccgaccgctgcgccttatccggtaactatcgtcttgagtccaacccggtaagacacgacttatcgccactggcagcagccactggtaacaggattagcagagcgaggtatgtaggcggtgctacagagttcttgaagtggtggcctaactacggctacactagaaggacagtatttggtatctgcgctctgctgaagccagttaccttcggaaaaagagttggtagctcttgatccggcaaacaaaccaccgctggtagcggtggtttttttgtttgcaagcagcagattacgcgcagaaaaaaaggatctcaagaagatcctttgatcttttctacggggtctgacgctcagtggaacgaaaactcacgttaagggattttggtcatgagattatcaaaaaggatcttcacctagatccttttaaattaaaaatgaagttttaaatcaatctaaagtatatatgagtaaacttggtctgacagttaccaatgcttaatcagtgaggcacctatctcagcgatctgtctatttcgttcatccatagttgcctgactccccgtcgtgtagataactacgatacgggagggcttaccatctggccccagtgctgcaatgataccgcgagacccacgctcaccggctccagatttatcagcaataaaccagccagccggaagggccgagcgcagaagtggtcctgcaactttatccgcctccatccagtctattaattgttgccgggaagctagagtaagtagttcgccagttaatagtttgcgcaacgttgttgccattgctacaggcatcgtggtgtcacgctcgtcgtttggtatggcttcattcagctccggttcccaacgatcaaggcgagttacatgatcccccatgttgtgcaaaaaagcggttagctccttcggtcctccgatcgttgtcagaagtaagttggccgcagtgttatcactcatggttatggcagcactgcataattctcttactgtcatgccatccgtaagatgcttttctgtgactggtgagtactcaaccaagtcattctgagaatagtgtatgcggcgaccgagttgctcttgcccggcgtcaatacgggataataccgcgccacatagcagaactttaaaagtgctcatcattggaaaacgttcttcggggcgaaaactctcaaggatcttaccgctgttgagatccagttcgatgtaacccactcgtgcacccaactgatcttcagcatcttttactttcaccagcgtttctgggtgagcaaaaacaggaaggcaaaatgccgcaaaaaagggaataagggcgacacggaaatgttgaatactcatactcttcctttttcaatattattgaagcatttatcagggttattgtctcatgagcggatacatatttgaatgtatttagaaaaataaacaaataggggttccgcgcacatttccccgaaaagtg".upper()
assert ES7_SEQUENCE.count("N") == 90  # 70 in exon, 20 in barcode

# Replace barcode with A's
assert ES7_SEQUENCE[4582:4602] == "N" * 20
BARCODE_SEQUENCE = "A" * 20
ES7_SEQUENCE = ES7_SEQUENCE[:4582] + BARCODE_SEQUENCE + ES7_SEQUENCE[4602:]

# Extract flanks: upstream ends with ...CCAGG|GTT, downstream starts with CAG|GTCTGAC...
# The variable exon occupies positions 3518..3588 (70nt of N's)
UPSTREAM_FLANK = ES7_SEQUENCE[:3518]   # ends with GTT
DOWNSTREAM_FLANK = ES7_SEQUENCE[3588:] # starts with CAG

# Sanity checks
assert UPSTREAM_FLANK[-3:] == "GTT", f"Expected GTT, got {UPSTREAM_FLANK[-3:]}"
assert DOWNSTREAM_FLANK[:3] == "CAG", f"Expected CAG, got {DOWNSTREAM_FLANK[:3]}"
assert ES7_SEQUENCE[3518:3588] == "N" * 70


#  INTERPRETABLE SPLICING MODEL
def score_one_exon_interpretable(exon_seq: str) -> Dict[str, Any]:
    """
    Should return:
    {
        "psi": 0.72
    }
    """
    exon = exon_seq.strip().upper().replace("T", "U")
    if not exon:
        raise ValueError("Please paste an exon sequence first.")
    if any(ch not in "ACGU" for ch in exon):
        raise ValueError("Only A/C/G/U are allowed (RNA).")

    j = get_vis_data(
        exon=exon,
        threshold=.001,
        use_new_grouping=True,
        dataset_name="ES7",
    )
    return {"psi": round(j['predicted_psi'],2)}


# SPLICE AI

def one_hot_encode_fixed(seq: str) -> np.ndarray:
    seq = seq.upper()
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    try:
        return np.array([mapping[base] for base in seq], dtype=np.float32)
    except KeyError as e:
        raise ValueError(f"Invalid base in sequence: {e.args[0]}")


def score_one_exon_spliceai(models, exon_seq, upstream_flank, downstream_flank):
    """
    Score one exon inserted into the construct and return
    acceptor, donor, and PSI proxy value.
    """
    exon_seq = exon_seq.upper().strip().replace("U", "T")
    construct = upstream_flank + exon_seq + downstream_flank

    #pad for SpliceAI
    padded_seq = "N" * 5000 + construct + "N" * 5000
    one_hot = np.expand_dims(one_hot_encode_fixed(padded_seq), axis=0)

    #average across all 5 SpliceAI models
    all_scores = []
    for model in models:
        preds = model.predict(one_hot, verbose=0)
        all_scores.append(preds)

    scores = np.mean(all_scores, axis=0)   # shape: (1, seq_len, 3)
    scores = scores[0]                     # shape: (seq_len, 3)

    #positions in construct coordinates
    acceptor_pos = 3515
    donor_pos = 3518 + len(exon_seq) + 2

    acceptor_score = float(scores[acceptor_pos, 1])
    donor_score = float(scores[donor_pos, 2])

    psi_proxy_avg = (acceptor_score + donor_score) / 2.0
    psi_proxy_prod = acceptor_score * donor_score

    return {
        "spliceai_acceptor": acceptor_score,
        "spliceai_donor": donor_score,
        "psi_proxy_avg": psi_proxy_avg,
        "psi_proxy_prod": psi_proxy_prod,
    }

# ALPHAGENOME
# https://github.com/google-deepmind/alphagenome
# Set: export ALPHAGENOME_API_KEY='your-key'

_dna_model = None
def get_dna_model():
    global _dna_model
    if _dna_model is None:
        key = os.environ.get("ALPHAGENOME_API_KEY")
        if not key:
            raise RuntimeError("set ALPHAGENOME_API_KEY in your environment (export ALPHAGENOME_API_KEY=...).")
        _dna_model = dna_client.create(key)
    return _dna_model


HELA_S3_ONTOLOGY = "EFO:0002791"
USAGE_HELA_S3_NAME = "usage_EFO:0002791 polyA plus RNA-seq"

AG_TOTAL_LEN = 16384  # AlphaGenome input length
UPSTREAM_DONOR_POS = 3388  # Exon 1 5'SS in construct coords
DOWNSTREAM_ACCEPTOR_OFFSET = 853  # Offset in DOWNSTREAM_FLANK to exon 3 3'SS


def get_ag_splice_site_prob(pred, padding_len, acceptor_pos, donor_pos):
    """Extract splice site probabilities at canonical acceptor and donor."""
    ss = pred.splice_sites.values[padding_len:AG_TOTAL_LEN - padding_len]
    return float(ss[acceptor_pos, 1]), float(ss[donor_pos, 0])


def get_ag_splice_site_usage(pred, padding_len, acceptor_pos, donor_pos):
    """Extract splice site usage at canonical acceptor and donor (HeLa S3)."""
    usage = pred.splice_site_usage.values[padding_len:AG_TOTAL_LEN - padding_len]
    usage_track_idx = pred.splice_site_usage.metadata[
        (pred.splice_site_usage.metadata["name"] == USAGE_HELA_S3_NAME) &
        (pred.splice_site_usage.metadata["strand"] == "+")
    ].index[0]
    return float(usage[acceptor_pos, usage_track_idx]), float(usage[donor_pos, usage_track_idx])


def get_ag_junction_psi(pred, include_j1, include_j2, skip_j):
    """Extract junction counts and compute PSI."""
    junction_idx = pred.splice_junctions.metadata.loc[
        pred.splice_junctions.metadata["ontology_curie"] == HELA_S3_ONTOLOGY
    ].index[0]

    def get_count(junction):
        idxs = np.where(pred.splice_junctions.junctions == junction)[0]
        return float(pred.splice_junctions.values[idxs[0], junction_idx]) if len(idxs) > 0 else np.nan

    incl1, incl2, skip = get_count(include_j1), get_count(include_j2), get_count(skip_j)

    # Compute PSI with imputation for missing junctions
    if np.isnan(incl1) and np.isnan(incl2):
        psi = 0.0 if not np.isnan(skip) else np.nan
    elif np.isnan(skip):
        psi = 1.0
    else:
        incl_min = np.nanmin([incl1, incl2])
        total = incl_min + skip
        psi = float(incl_min / total) if total > 0 else np.nan

    return incl1, incl2, skip, psi


def score_one_exon_alphagenome(exon_seq: str):
    exon_seq = exon_seq.upper().strip().replace("U", "T")
    exon_len = len(exon_seq)

    construct = UPSTREAM_FLANK + exon_seq + DOWNSTREAM_FLANK
    construct_len = len(construct)
    padding_len = (AG_TOTAL_LEN - construct_len) // 2

    if construct_len > AG_TOTAL_LEN:
        raise ValueError(
            f"Construct length {construct_len} exceeds AlphaGenome input length {AG_TOTAL_LEN}."
        )

    #splice site positions in construct coordinates
    acceptor_pos = 3515
    donor_pos = 3518 + exon_len + 2

    #junction positions in padded sequence coordinates
    ud = UPSTREAM_DONOR_POS + padding_len
    acc = acceptor_pos + padding_len
    don = (3518 + exon_len + 3) + padding_len
    da = (3518 + exon_len + DOWNSTREAM_ACCEPTOR_OFFSET) + padding_len

    include_j1 = genome.Interval(chromosome="", start=ud, end=acc, strand="+", name="")
    include_j2 = genome.Interval(chromosome="", start=don, end=da, strand="+", name="")
    skip_j = genome.Interval(chromosome="", start=ud, end=da, strand="+", name="")

    interval = genome.Interval(
        chromosome="chrPlasmid",
        start=-padding_len,
        end=AG_TOTAL_LEN - padding_len,
        strand="+",
    )

    padded_seq = construct.center(AG_TOTAL_LEN, "N")

    preds = get_dna_model().predict_sequences(
        sequences=[padded_seq],
        requested_outputs=[
            dna_client.OutputType.SPLICE_SITES,
            dna_client.OutputType.SPLICE_SITE_USAGE,
            dna_client.OutputType.SPLICE_JUNCTIONS,
        ],
        ontology_terms=[HELA_S3_ONTOLOGY],
        max_workers=1,
        intervals=[interval],
    )

    pred = preds[0]

    p_acc, p_don = get_ag_splice_site_prob(pred, padding_len, acceptor_pos, donor_pos)
    u_acc, u_don = get_ag_splice_site_usage(pred, padding_len, acceptor_pos, donor_pos)
    incl1, incl2, skip, psi = get_ag_junction_psi(pred, include_j1, include_j2, skip_j)

    return {
        "alphagenome_acceptor_prob": p_acc,
        "alphagenome_donor_prob": p_don,
        "alphagenome_acceptor_usage": u_acc,
        "alphagenome_donor_usage": u_don,
        "alphagenome_inclusion_junction_1": incl1,
        "alphagenome_inclusion_junction_2": incl2,
        "alphagenome_skip_junction": skip,
        "alphagenome_psi": psi,
    }

# HAL (in progress)



# Max Ent Scan

_ment_path = os.path.join(REPO_ROOT, "models", "splice_site_scoring")
if _ment_path not in sys.path:
    sys.path.insert(0, _ment_path)

matrix5 = maxent.load_matrix5()
matrix3 = maxent.load_matrix3()

FLANKING_NT = 25
UPSTREAM_CONTEXT = UPSTREAM_FLANK[-FLANKING_NT:]
DOWNSTREAM_CONTEXT = DOWNSTREAM_FLANK[:FLANKING_NT]

def score_5ss(nts):
    """Score a 9nt 5'SS window (3 exonic + 6 intronic)."""
    assert len(nts) == 9
    return maxent.score5(nts, matrix5)

def score_3ss(nts):
    """Score a 23nt 3'SS window (20 intronic + 3 exonic)."""
    assert len(nts) == 23
    return maxent.score3(nts, matrix3)

def scan_5ss(seq):
    """Scan all 9-mer windows and return array of (position, score)."""
    return np.array([(i, score_5ss(seq[i:i+9])) for i in range(len(seq) - 8)])

def scan_3ss(seq):
    """Scan all 23-mer windows and return array of (position, score)."""
    return np.array([(i, score_3ss(seq[i:i+23])) for i in range(len(seq) - 22)])

def max_cryptic_strength(scores_array, canonical_idx):
    """Return the max score excluding the canonical splice site position."""
    mask = np.zeros(scores_array.shape[0], dtype=bool)
    mask[canonical_idx] = True
    values = np.ma.masked_array(scores_array[:, 1], mask=mask)
    return float(values.max())

def logistic(x, k=0.4, x0=6.0):
    """Compress MaxEnt-like scores to 0-1."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def score_one_exon_maxent(exon_seq):
    """
    Score one exon in the ES7 construct with MaxEntScan and return
    acceptor, donor, and PSI values.
    """
    exon_seq = exon_seq.upper().strip().replace("U", "T")

    #build local sequence around exon using 25 nt flanks
    local_seq = UPSTREAM_CONTEXT + exon_seq + DOWNSTREAM_CONTEXT

    #canonical windows
    canonical_3ss_pos = 2
    canonical_5ss_pos = FLANKING_NT + len(exon_seq)

    acceptor_23mer = local_seq[canonical_3ss_pos:canonical_3ss_pos + 23]
    donor_9mer = local_seq[canonical_5ss_pos:canonical_5ss_pos + 9]

    acceptor_score = score_3ss(acceptor_23mer)
    donor_score = score_5ss(donor_9mer)

    #transform raw MaxEnt scores to 0-1 proxies
    acceptor_norm = logistic(acceptor_score)
    donor_norm = logistic(donor_score)

    psi_proxy_avg = (acceptor_norm + donor_norm) / 2.0
    psi_proxy_prod = acceptor_norm * donor_norm

    return {
        "maxent_acceptor_raw": float(acceptor_score),
        "maxent_donor_raw": float(donor_score),
        "maxent_acceptor_norm": float(acceptor_norm),
        "maxent_donor_norm": float(donor_norm),
        "maxent_psi_proxy_avg": float(psi_proxy_avg), 
        "maxent_psi_proxy_prod": float(psi_proxy_prod), ##use this for proxy
        "acceptor_23mer": acceptor_23mer,
        "donor_9mer": donor_9mer,
    }

def add_result(model, predicted_psi, actual_psi):
    return {
        "model": model,
        "psi_score": predicted_psi,
        "abs_error": safe_abs_error(predicted_psi, actual_psi),
        "signed_error": safe_signed_error(predicted_psi, actual_psi),
        "actual_psi": actual_psi,
    }


def compare_models(exon_seq, actual_psi, spliceai_models):
    """Run all four predictors for one ES7 exon and return a results table."""
    result_list = []

    try:
        interpretable = score_one_exon_interpretable(exon_seq)
        result_list.append(
            add_result("Interpretable Splicing Model", interpretable["psi"], actual_psi)
        )
    except Exception as e:
        traceback.print_exc()
        result_list.append({"model": "Interpretable Splicing Model", "error": str(e)})

    try:
        spliceai = score_one_exon_spliceai(
            spliceai_models, exon_seq, UPSTREAM_FLANK, DOWNSTREAM_FLANK
        )
        result_list.append(add_result("SpliceAI", spliceai["psi_proxy_prod"], actual_psi))
    except Exception as e:
        result_list.append({"model": "SpliceAI", "error": str(e)})

    try:
        alphagenome = score_one_exon_alphagenome(exon_seq)
        result_list.append(
            add_result("AlphaGenome", alphagenome["alphagenome_psi"], actual_psi)
        )
    except Exception as e:
        result_list.append({"model": "AlphaGenome", "error": str(e)})

    try:
        maxentscan = score_one_exon_maxent(exon_seq)
        result_list.append(
            add_result("MaxEntScan", maxentscan["maxent_psi_proxy_prod"], actual_psi)
        )
    except Exception as e:
        result_list.append({"model": "MaxEntScan", "error": str(e)})

    return pd.DataFrame(result_list)


_spliceai_paths = [os.path.join(REPO_ROOT, f"models/spliceai{i}.h5") for i in range(1, 6)]
spliceai_models = [load_model(p) for p in _spliceai_paths]


if __name__ == "__main__":
    from datetime import datetime

    # --- edit these when running the script ---
    EXON = "GGTAGTACGCCAATTCGCCGGTGCCGCGAGCCAGAGGCTACCAAAACTTGACAAGCCTACATATACTACT"
    ACTUAL_PSI = 0.963
    # ------------------------------------------

    df = compare_models(EXON, ACTUAL_PSI, spliceai_models)

    print()
    print("  ES7 exon comparison")
    print("  " + "-" * 56)
    print(df.to_string(index=False))
    print()

    out_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"psi_comparison_{stamp}.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print()
