import pandas as pd
from . import rasta
import numpy as np


def extract_speech_features(signal, sr=48000):
    # ====
    # MFCC
    # ====

    # mfcc = rasta.melfcc(signal, fs=sr, max_freq=8000, n_mfcc=25, n_bands=24,
    #     fbtype='fcmel', dcttype=1, usecmp=True, window_time=0.025, hop_time=0.010,
    #     preemph=0, dither=1)

    mfcc = rasta.melfcc(
        signal,
        fs=sr,
        max_freq=2000,
        n_mfcc=13,
        n_bands=12,
        fbtype="fcmel",
        dcttype=1,
        usecmp=True,
        window_time=0.025,
        hop_time=0.010,
        preemph=0,
        dither=1,
    )

    mfcc_delta = rasta.deltas(mfcc)
    mfcc_delta2 = rasta.deltas(rasta.deltas(mfcc, w=5), w=5)

    # MFCC mean/std
    mfcc_avg = mfcc.mean(axis=1)
    mfcc_delta_avg = mfcc_delta.mean(axis=1)
    mfcc_delta2_avg = mfcc_delta2.mean(axis=1)

    mfcc_std = mfcc.std(axis=1)
    mfcc_delta_std = mfcc_delta.std(axis=1)
    mfcc_delta2_std = mfcc_delta2.std(axis=1)

    # MFCC max/min
    row_indexes = np.arange(mfcc.shape[0], dtype=int)
    argmax = np.argmax(mfcc, axis=1)
    max_indexes = np.array((row_indexes, argmax))
    mfcc_max_values = mfcc[tuple(max_indexes)]
    mfcc_max_indexes = argmax / mfcc.shape[1]

    argmin = np.argmin(mfcc, axis=1)
    min_indexes = np.array((row_indexes, argmin))
    mfcc_min_values = mfcc[tuple(min_indexes)]
    mfcc_min_indexes = argmin / mfcc.shape[1]

    # MFCC_DELTA max/min
    argmax = np.argmax(mfcc_delta, axis=1)
    max_indexes = np.array((row_indexes, argmax))
    mfcc_delta_max_values = mfcc_delta[tuple(max_indexes)]
    mfcc_delta_max_indexes = argmax / mfcc_delta.shape[1]

    argmin = np.argmin(mfcc_delta, axis=1)
    min_indexes = np.array((row_indexes, argmin))
    mfcc_delta_min_values = mfcc_delta[tuple(min_indexes)]
    mfcc_delta_min_indexes = argmin / mfcc_delta.shape[1]

    # MFCC_DELTA2 max/min
    argmax = np.argmax(mfcc_delta2, axis=1)
    max_indexes = np.array((row_indexes, argmax))
    mfcc_delta2_max_values = mfcc_delta2[tuple(max_indexes)]
    mfcc_delta2_max_indexes = argmax / mfcc_delta2.shape[1]

    argmin = np.argmin(mfcc_delta2, axis=1)
    min_indexes = np.array((row_indexes, argmin))
    mfcc_delta2_min_values = mfcc_delta2[tuple(min_indexes)]
    mfcc_delta2_min_indexes = argmin / mfcc_delta2.shape[1]

    # MFCC poly 2 - xˆ2 part only
    col_indexes = np.arange(mfcc.shape[1], dtype=int)
    mfcc_curvature_poly2 = np.polyfit(col_indexes, mfcc.T, 2).T[:, 0]
    mfcc_delta_curvature_poly2 = np.polyfit(col_indexes, mfcc_delta.T, 2).T[:, 0]
    mfcc_delta2_curvature_poly2 = np.polyfit(col_indexes, mfcc_delta2.T, 2).T[:, 0]

    # MFCC poly 1
    mfcc_poly1 = np.polyfit(col_indexes, mfcc.T, 1).T
    mfcc_delta_poly1 = np.polyfit(col_indexes, mfcc_delta.T, 1).T
    mfcc_delta2_poly1 = np.polyfit(col_indexes, mfcc_delta2.T, 1).T

    # MFCC poly 1 - slope
    mfcc_slope_poly1 = mfcc_poly1[:, 0]
    mfcc_delta_slope_poly1 = mfcc_delta_poly1[:, 0]
    mfcc_delta2_slope_poly1 = mfcc_delta2_poly1[:, 0]

    # MFCC poly 1 - offset
    mfcc_offset_poly1 = mfcc_poly1[:, 1]
    mfcc_delta_offset_poly1 = mfcc_delta_poly1[:, 1]
    mfcc_delta2_offset_poly1 = mfcc_delta2_poly1[:, 1]

    # MFCC Zero Crossing Rate
    mfcc_ZCR = rasta.ZCR(mfcc)
    mfcc_delta_ZCR = rasta.ZCR(mfcc_delta)
    mfcc_delta2_ZCR = rasta.ZCR(mfcc_delta2)

    # =========
    # RASTA PLP
    # =========
    rplp = rasta.rastaplp(
        signal, fs=sr, win_time=0.025, hop_time=0.010, dorasta=True, modelorder=12
    )
    rplp_delta = rasta.deltas(rplp)
    rplp_delta2 = rasta.deltas(rasta.deltas(rplp, w=5), w=5)

    # RASTA PLP mean/std
    rplp_avg = rplp.mean(axis=1)
    rplp_delta_avg = rplp_delta.mean(axis=1)
    rplp_delta2_avg = rplp_delta2.mean(axis=1)

    rplp_std = rplp.std(axis=1)
    rplp_delta_std = rplp_delta.std(axis=1)
    rplp_delta2_std = rplp_delta2.std(axis=1)

    # RASTA max/min
    row_indexes = np.arange(rplp.shape[0], dtype=int)
    argmax = np.argmax(rplp, axis=1)
    max_indexes = np.array((row_indexes, argmax))
    rplp_max_values = rplp[tuple(max_indexes)]
    rplp_max_indexes = argmax / rplp.shape[1]

    argmin = np.argmin(rplp, axis=1)
    min_indexes = np.array((row_indexes, argmin))
    rplp_min_values = rplp[tuple(min_indexes)]
    rplp_min_indexes = argmin / rplp.shape[1]

    # RASTA_DELTA max/min
    argmax = np.argmax(rplp_delta, axis=1)
    max_indexes = np.array((row_indexes, argmax))
    rplp_delta_max_values = rplp_delta[tuple(max_indexes)]
    rplp_delta_max_indexes = argmax / rplp_delta.shape[1]

    argmin = np.argmin(rplp_delta, axis=1)
    min_indexes = np.array((row_indexes, argmin))
    rplp_delta_min_values = rplp_delta[tuple(min_indexes)]
    rplp_delta_min_indexes = argmin / rplp_delta.shape[1]

    # RASTA_DELTA2 max/min
    argmax = np.argmax(rplp_delta2, axis=1)
    max_indexes = np.array((row_indexes, argmax))
    rplp_delta2_max_values = rplp_delta2[tuple(max_indexes)]
    rplp_delta2_max_indexes = argmax / rplp_delta2.shape[1]

    argmin = np.argmin(rplp_delta2, axis=1)
    min_indexes = np.array((row_indexes, argmin))
    rplp_delta2_min_values = rplp_delta2[tuple(min_indexes)]
    rplp_delta2_min_indexes = argmin / rplp_delta2.shape[1]

    # RASTA poly 2 - xˆ2 part only
    col_indexes = np.arange(rplp.shape[1], dtype=int)
    rplp_curvature_poly2 = np.polyfit(col_indexes, rplp.T, 2).T[:, 0]
    rplp_delta_curvature_poly2 = np.polyfit(col_indexes, rplp_delta.T, 2).T[:, 0]
    rplp_delta2_curvature_poly2 = np.polyfit(col_indexes, rplp_delta2.T, 2).T[:, 0]

    # RASTA poly 1
    rplp_poly1 = np.polyfit(col_indexes, rplp.T, 1).T
    rplp_delta_poly1 = np.polyfit(col_indexes, rplp_delta.T, 1).T
    rplp_delta2_poly1 = np.polyfit(col_indexes, rplp_delta2.T, 1).T

    # RASTA poly 1 - slope
    rplp_slope_poly1 = rplp_poly1[:, 0]
    rplp_delta_slope_poly1 = rplp_delta_poly1[:, 0]
    rplp_delta2_slope_poly1 = rplp_delta2_poly1[:, 0]

    # RASTA poly 1 - offset
    rplp_offset_poly1 = rplp_poly1[:, 1]
    rplp_delta_offset_poly1 = rplp_delta_poly1[:, 1]
    rplp_delta2_offset_poly1 = rplp_delta2_poly1[:, 1]

    # RASTA Zero Crossing Rate
    rplp_ZCR = rasta.ZCR(rplp)
    rplp_delta_ZCR = rasta.ZCR(rplp_delta)
    rplp_delta2_ZCR = rasta.ZCR(rplp_delta2)

    # ========
    # ASSEMBLY
    # ========
    labels = lambda prefix, postfix, matrix: [
        f"{prefix}[{x + 1}]_{postfix}" for x in range(matrix.shape[0])
    ]
    row = np.array([])
    label_list = []

    # MFCC
    sets = [
        ("mean", mfcc_avg),
        ("de_mean", mfcc_delta_avg),
        ("dede_mean", mfcc_delta2_avg),
        ("std", mfcc_std),
        ("de_std", mfcc_delta_std),
        ("dede_std", mfcc_delta2_std),
        ("curvature_poly2", mfcc_curvature_poly2),
        ("de_curvature_poly2", mfcc_delta_curvature_poly2),
        ("dede_curvature_poly2", mfcc_delta2_curvature_poly2),
        ("slope_poly1", mfcc_slope_poly1),
        ("de_slope_poly1", mfcc_delta_slope_poly1),
        ("dede_slope_poly1", mfcc_delta2_slope_poly1),
        ("offset_poly1", mfcc_offset_poly1),
        ("de_offset_poly1", mfcc_delta_offset_poly1),
        ("dede_offset_poly1", mfcc_delta2_offset_poly1),
        ("max", mfcc_max_values),
        ("de_max", mfcc_delta_max_values),
        ("dede_max", mfcc_delta2_max_values),
        ("relmaxpos", mfcc_max_indexes),
        ("de_relmaxpos", mfcc_delta_max_indexes),
        ("dede_relmaxpos", mfcc_delta2_max_indexes),
        ("min", mfcc_min_values),
        ("de_min", mfcc_delta_min_values),
        ("dede_min", mfcc_delta2_min_values),
        ("relminpos", mfcc_min_indexes),
        ("de_relminpos", mfcc_delta_min_indexes),
        ("dede_relminpos", mfcc_delta2_min_indexes),
        ("ZCR", mfcc_ZCR),
        ("de_ZCR", mfcc_delta_ZCR),
        ("dede_ZCR", mfcc_delta2_ZCR),
    ]
    for postf, dataset in sets:
        label_list += labels("mfcc", postf, dataset)
        row = np.concatenate((row, dataset), axis=0)

    # RASTA
    sets = [
        ("mean", rplp_avg),
        ("de_mean", rplp_delta_avg),
        ("dede_mean", rplp_delta2_avg),
        ("std", rplp_std),
        ("de_std", rplp_delta_std),
        ("dede_std", rplp_delta2_std),
        ("curvature_poly2", rplp_curvature_poly2),
        ("de_curvature_poly2", rplp_delta_curvature_poly2),
        ("dede_curvature_poly2", rplp_delta2_curvature_poly2),
        ("slope_poly1", rplp_slope_poly1),
        ("de_slope_poly1", rplp_delta_slope_poly1),
        ("dede_slope_poly1", rplp_delta2_slope_poly1),
        ("offset_poly1", rplp_offset_poly1),
        ("de_offset_poly1", rplp_delta_offset_poly1),
        ("dede_offset_poly1", rplp_delta2_offset_poly1),
        ("max", rplp_max_values),
        ("de_max", rplp_delta_max_values),
        ("dede_max", rplp_delta2_max_values),
        ("relmaxpos", rplp_max_indexes),
        ("de_relmaxpos", rplp_delta_max_indexes),
        ("dede_relmaxpos", rplp_delta2_max_indexes),
        ("min", rplp_min_values),
        ("de_min", rplp_delta_min_values),
        ("dede_min", rplp_delta2_min_values),
        ("relminpos", rplp_min_indexes),
        ("de_relminpos", rplp_delta_min_indexes),
        ("dede_relminpos", rplp_delta2_min_indexes),
        ("ZCR", rplp_ZCR),
        ("de_ZCR", rplp_delta_ZCR),
        ("dede_ZCR", rplp_delta2_ZCR),
    ]

    for postf, dataset in sets:
        label_list += labels("rplpc", postf, dataset)
        row = np.concatenate((row, dataset), axis=0)

    return dict(zip(label_list, row))
