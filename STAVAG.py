# -*- coding: utf-8 -*-
"""
Created on April 2 18:59:29 2025

@author: Qunlun Shen
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def generate_coord_dict(n_dim: int) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Build a placeholder dict for coordinate axes.

    Args:
        n_dim: Number of coordinate dimensions. Supported up to four.

    Returns:
        A dict mapping axis names to None. For example
        {'x': None, 'y': None} for two dimensions.

    Raises:
        ValueError: If n_dim is greater than four.
    """
    if n_dim > 4:
        raise ValueError("Only up to n_dim=4 is supported. Please provide a coords DataFrame with less than four columns.")
    
    keys = ['x', 'y', 'z', 'a'][:n_dim]
    return {key: None for key in keys}

def calculate_sps(
    coord_dict_raw: Dict[str, pd.DataFrame],
    coord_dict_rand: Dict[str, np.ndarray],
    n_dim: int,
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute STAVAG priority scores (sps) for each axis.

    The score is the right tail proportion of random importances
    that are greater than or equal to the observed importance.

    Args:
        coord_dict_raw: Dict of DataFrames per axis. Each DataFrame
            must contain columns 'Feature' and 'Importance'.
        coord_dict_rand: Dict of random importance arrays per axis.
        n_dim: Number of coordinate dimensions.
        keys: Optional explicit axis names to use.

    Returns:
        The same dict as coord_dict_raw with a new column 'sps'
        added to each axis DataFrame.
    """
    if keys:
        pass
    else:
        keys = ['x', 'y', 'z', 'a'][:n_dim]
    for k in range(n_dim):
        coord_dict_raw[keys[k]]['sps'] = [(np.sum(coord_dict_rand[keys[k]] >= val) +1) / (len(coord_dict_rand[keys[k]]) +1) for val in coord_dict_raw[keys[k]]['Importance']]
    return coord_dict_raw

def keep_variant_genes(
    coord_dict_raw: Dict[str, pd.DataFrame],
    coord_dict_rand: Dict[str, np.ndarray],
    n_dim: int,
    threshold: float = 0.05,
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Filter genes whose observed importance exceeds a random baseline.

    For each axis this keeps rows where Importance is greater than
    a high percentile of the random importance distribution.

    Args:
        coord_dict_raw: Dict of DataFrames per axis with importance values.
        coord_dict_rand: Dict of random importance arrays per axis.
        n_dim: Number of coordinate dimensions.
        threshold: Significance level. For example 0.05 targets the top
            tail of the random distribution.
        keys: Optional explicit axis names.

    Returns:
        Filtered dict with the same structure as coord_dict_raw.
    """
    if keys:
        pass
    else:
        keys = ['x', 'y', 'z', 'a'][:n_dim]
    for k in range(n_dim):
        top_percentile = np.percentile(coord_dict_rand[keys[k]], 100*(1-threshold))
        coord_dict_raw[keys[k]] = coord_dict_raw[keys[k]][coord_dict_raw[keys[k]]['Importance']>top_percentile]
    return coord_dict_raw

def DVG_detection(
    adata: sc.AnnData,
    coords: np.ndarray,
    sps: bool = False,
    threshold: float = 0.05,
    num_perm: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Detect Directionally Variable Genes (DVGs) using regression on spatial coordinates.

    Args:
        adata (AnnData): AnnData with expression matrix ``adata.X`` and gene names
            ``adata.var.index``.
        coords (ndarray): Spatial coordinates of cells with shape ``(n_cells, n_dim)``.
            For example two columns for x and y or three columns for x y z.
        sps (bool, optional): If True, compute STAVAG priority scores by comparing
            observed importances with random baselines. Defaults to False.
        threshold (float, optional): Importance threshold used when selecting DVGs.
            Larger values keep more genes. Defaults to 0.05.
        num_perm (int, optional): Number of permutations used to build an empirical
            null distribution of feature importances.
            - If 1: keep the original single-permutation behavior.
            - If > 1: must be >= 100; empirical p-values are computed for each gene.

    Returns:
        Dict[str, DataFrame]: Dictionary containing top important genes per coordinate axis (e.g., 'x', 'y', 'z'),
        filtered with the threshold.

        - For num_perm == 1: same structure as before (with SPS scores if sps=True).
        - For num_perm > 1: each DataFrame contains columns ['Feature', 'Importance', 'null_mean', 'pval'] and is filtered by p-value (<= threshold).
    """
    np.random.seed(0)
    n_dim = coords.shape[1]
    if n_dim == 1:
        raise ValueError("n_dim must be at least 2.")

    # Axis labels based on dimensionality
    keys = ['x', 'y', 'z', 'a'][:n_dim]

    X = adata.X          # Gene expression matrix
    Y = coords.copy()    # Spatial coordinates

    # LightGBM regression parameters
    params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        objective="regression",
        metric="mse",
        boosting_type="gbdt",
        colsample_bytree=0.2,
        subsample=0.9,
        subsample_freq=5,
        importance_type='gain',
        verbosity=-1,
    )

    lgb_model = lgb.LGBMRegressor(**params)

    # ===== Case 1: num_perm == 1 (original behavior with a single random baseline) =====
    if num_perm == 1:
        # Initialize coordinate-wise containers
        coord_dict_raw = generate_coord_dict(n_dim)
        coord_dict_rand = generate_coord_dict(n_dim)

        # Fit MultiOutput LightGBM on real data
        model = MultiOutputRegressor(lgb_model)
        model.fit(X, Y)

        # Observed feature importances per coordinate axis
        feature_importances = []
        for i, estimator in enumerate(model.estimators_):
            imp_df = pd.DataFrame({
                'Feature': adata.var.index,
                'Importance': estimator.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            feature_importances.append(imp_df)

        for k in range(n_dim):
            coord_dict_raw[keys[k]] = feature_importances[k]

        # Build a single random baseline by permuting rows of X
        num_rows = X.shape[0]
        shuffled_indices = np.random.permutation(num_rows)
        random_matrix = X[shuffled_indices, :]

        lgb_model2 = lgb.LGBMRegressor(**params)
        model2 = MultiOutputRegressor(lgb_model2)
        model2.fit(random_matrix, Y)

        # Random-baseline feature importances
        for k in range(n_dim):
            coord_dict_rand[keys[k]] = model2.estimators_[k].feature_importances_

        # Compare observed vs random importances
        if sps:
            # Compute STAVAG priority scores
            coord_dict_raw = calculate_sps(coord_dict_raw, coord_dict_rand, n_dim)
        else:
            # Keep variant genes based on importance threshold
            coord_dict_raw = keep_variant_genes(
                coord_dict_raw, coord_dict_rand, n_dim, threshold=threshold
            )
        return coord_dict_raw

    # ===== Case 2: num_perm > 1 (empirical null + permutation p-values) =====
    if num_perm < 100:
        raise ValueError(
            f"num_perm must be >= 100 when using multiple permutations; got {num_perm}."
        )

    # 1) Fit MultiOutput LightGBM on real data to obtain observed importances
    model = MultiOutputRegressor(lgb_model)
    model.fit(X, Y)

    n_genes = X.shape[1]

    # obs_importances[axis_key] has shape (n_genes,)
    obs_importances: Dict[str, np.ndarray] = {}
    for k, axis_key in enumerate(keys):
        obs_importances[axis_key] = model.estimators_[k].feature_importances_.astype(float)

    # 2) Multiple permutations: permute rows of X, refit model, store null importances
    # null_importances[axis_key] has shape (num_perm, n_genes)
    null_importances: Dict[str, np.ndarray] = {
        axis_key: np.zeros((num_perm, n_genes), dtype=float) for axis_key in keys
    }

    num_rows = X.shape[0]
    for b in tqdm_notebook(range(num_perm)):
        shuffled_indices = np.random.permutation(num_rows)
        random_matrix = X[shuffled_indices, :]

        perm_model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
        perm_model.fit(random_matrix, Y)

        for k, axis_key in enumerate(keys):
            null_importances[axis_key][b, :] = perm_model.estimators_[k].feature_importances_

    # 3) Compute empirical permutation p-values and filter genes
    coord_results: Dict[str, pd.DataFrame] = {}

    for axis_key in keys:
        obs = obs_importances[axis_key]         # shape: (n_genes,)
        null_mat = null_importances[axis_key]   # shape: (num_perm, n_genes)

        # Empirical p-value: P(null >= observed)
        # p_j = (1 + count(null_b >= obs_j)) / (num_perm + 1)
        counts = (null_mat >= obs[None, :]).sum(axis=0)
        pvals = (counts + 1.0) / (num_perm + 1.0)

        null_mean = null_mat.mean(axis=0)

        df = pd.DataFrame({
            "Feature": adata.var.index,
            "Importance": obs,
            "null_mean": null_mean,
            "pval": pvals,
        })

        # Sort by p-value
        df = df.sort_values(by="pval", ascending=True)

        # Apply p-value threshold
        if threshold is not None:
            df = df[df["pval"] <= threshold]

        df = df.reset_index(drop=True)
        coord_results[axis_key] = df

    return coord_results

def TVG_detection(
    adata: sc.AnnData,
    coords: np.ndarray,
    sps: bool = False,
    threshold: float = 0.05,
    num_perm: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Detect Temporally Variable Genes (TVGs) using regression on a 1D time coordinate.

    Args:
        adata (AnnData): An AnnData object containing gene expression matrix
            ``adata.X`` and gene names ``adata.var.index``.
        coords (ndarray): 1D temporal coordinate of cells with shape ``(n_cells, 1)``.
        sps (bool, optional):
            If True and num_perm == 1, compute STAVAG priority scores by comparing
            observed importances with a single random baseline (original behavior).
            Defaults to False.
        threshold (float, optional):
            - If num_perm == 1: cutoff used by ``keep_variant_genes`` to select TVGs
              based on importance.
            - If num_perm > 1: p-value cutoff; genes with pval <= threshold are kept.
            Defaults to 0.05.
        num_perm (int, optional): Number of permutations used to build an empirical
            null distribution of feature importances.
            - If 1: keep the original single-permutation behavior.
            - If > 1: must be >= 100; empirical permutation p-values are computed
              for each gene.

    Returns:
        Dict[str, DataFrame]: Dictionary containing important genes over the time
        axis ``'T'``.

        - For num_perm == 1: same structure as before (with SPS scores
          if sps=True), already filtered by the given threshold.
        - For num_perm > 1: the DataFrame under key 'T' contains columns: ['Feature', 'Importance', 'null_mean', 'pval'] and is filtered by p-value (<= threshold).
    """
    np.random.seed(0)

    n_dim = coords.shape[1]
    if n_dim != 1:
        raise ValueError("n_dim must be 1 for TVG_detection.")

    keys = ['T']
    X = adata.X
    Y = coords.copy()

    # LightGBM regression parameters
    params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        objective="regression",
        metric="mse",
        boosting_type="gbdt",
        colsample_bytree=0.2,
        subsample=0.9,
        subsample_freq=5,
        importance_type='gain',
        verbosity=-1,
    )

    # ===== Case 1: num_perm == 1 (original behavior) =====
    if num_perm == 1:
        coord_dict_raw: Dict[str, pd.DataFrame] = {}
        coord_dict_rand: Dict[str, np.ndarray] = {}

        # Fit LightGBM on real data
        model = lgb.LGBMRegressor(**params)
        model.fit(X, Y)
        _ = model.predict(X)

        # Observed feature importances (gain)
        imp_df = pd.DataFrame({
            'Feature': adata.var.index,
            'Importance': model.booster_.feature_importance(importance_type='gain')
        }).sort_values(by='Importance', ascending=False)
        coord_dict_raw['T'] = imp_df

        # Build a single random baseline by permuting rows of X
        num_rows = X.shape[0]
        shuffled_indices = np.random.permutation(num_rows)
        random_matrix = X[shuffled_indices, :]

        model2 = lgb.LGBMRegressor(**params)
        model2.fit(random_matrix, Y)
        _ = model2.predict(random_matrix)

        coord_dict_rand['T'] = model2.booster_.feature_importance(importance_type='gain')

        # Compare observed vs random importances
        if sps:
            coord_dict_raw = calculate_sps(
                coord_dict_raw, coord_dict_rand, n_dim, keys=['T']
            )
        else:
            coord_dict_raw = keep_variant_genes(
                coord_dict_raw, coord_dict_rand, n_dim,
                threshold=threshold, keys=['T']
            )
        return coord_dict_raw

    # ===== Case 2: num_perm > 1 (empirical null + permutation p-values) =====
    if num_perm < 100:
        raise ValueError(
            f"num_perm must be >= 100 when using multiple permutations; got {num_perm}."
        )

    # 1) Fit LightGBM on real data to obtain observed importances
    base_model = lgb.LGBMRegressor(**params)
    base_model.fit(X, Y)
    _ = base_model.predict(X)

    n_genes = X.shape[1]

    # Observed importances (shape: (n_genes,))
    obs_importances = base_model.booster_.feature_importance(
        importance_type='gain'
    ).astype(float)

    # 2) Multiple permutations: permute rows of X and refit model to get null importances
    #    null_importances has shape (num_perm, n_genes)
    null_importances = np.zeros((num_perm, n_genes), dtype=float)

    num_rows = X.shape[0]
    for b in tqdm_notebook(range(num_perm)):
        shuffled_indices = np.random.permutation(num_rows)
        random_matrix = X[shuffled_indices, :]

        perm_model = lgb.LGBMRegressor(**params)
        perm_model.fit(random_matrix, Y)
        _ = perm_model.predict(random_matrix)

        null_importances[b, :] = perm_model.booster_.feature_importance(
            importance_type='gain'
        ).astype(float)

    # 3) Compute empirical permutation p-values for each gene
    #    p_j = (1 + count(null_b >= obs_j)) / (num_perm + 1)
    counts = (null_importances >= obs_importances[None, :]).sum(axis=0)
    pvals = (counts + 1.0) / (num_perm + 1.0)
    null_mean = null_importances.mean(axis=0)

    df = pd.DataFrame({
        "Feature": adata.var.index,
        "Importance": obs_importances,
        "null_mean": null_mean,
        "pval": pvals,
    })

    # Sort by p-value and apply cutoff
    df = df.sort_values(by="pval", ascending=True)
    if threshold is not None:
        df = df[df["pval"] <= threshold]
    df = df.reset_index(drop=True)

    # Return in the same Dict[str, DataFrame] format
    coord_results: Dict[str, pd.DataFrame] = {"T": df}
    return coord_results

def gene_modules(adata: sc.AnnData, gene_list: Sequence[str]) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Cluster genes into modules using correlation among selected genes.

    Args:
        adata (AnnData): AnnData that contains the expression matrix ``adata.X``
            and gene names in ``adata.var.index``.
        gene_list (Sequence[str]): Genes to include when building modules.
            Each gene should exist in ``adata.var.index``.

    Returns:
        Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:

            Z: Linkage matrix from hierarchical clustering.

            corr: Gene to gene correlation matrix as a pandas DataFrame.Index and columns are gene names in ``gene_list``.

            df: Expression matrix of the selected genes as a pandas DataFrame. Rows are cells and columns are genes.
    """
    df = adata[:, gene_list].to_df()
    corr = df.corr()
    Z = linkage(corr, 'complete', metric='correlation')
    return Z, corr, df