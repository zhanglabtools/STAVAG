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
    adata: AnnData,
    coords: np.ndarray,
    sps: bool = False,
    threshold: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """
    Detect Directionally Variable Genes (DVGs) using regression on spatial coordinates.

    Args:
    adata : AnnData
        An AnnData object containing gene expression matrix (adata.X) and gene names (adata.var.index).
    coords : np.ndarray
        Spatial coordinates of cells (shape: [n_cells, n_dimensions]; e.g., 2D [x,y] or 3D [x,y,z]).
    sps : bool, optional
        Whether to compute sps by comparing with random feature importances (default: False).

    Returns:
    coord_dict_raw : dict
        Dictionary containing top important genes per coordinate axis (e.g., 'x', 'y', 'z'),
        filtered by sps.
    """
    np.random.seed(0)
    n_dim = coords.shape[1]
    keys = ['x', 'y', 'z', 'a'][:n_dim] # Axis labels based on dimensionality
    X = adata.X  # Gene expression matrix
    Y = coords.copy() # Spatial coordinates

    # Regression parameters
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
    if n_dim == 1:
       raise ValueError("n_dim must be at least 2.")
    elif n_dim >= 2:
        # Initialize coordinate importance dictionaries
        coord_dict_raw = generate_coord_dict(n_dim)
        coord_dict_rand = generate_coord_dict(n_dim)
        # Train model on real data
        model = MultiOutputRegressor(lgb_model)
        model.fit(X, Y)
        Y_pred = model.predict(X)

        weights = []
        mse_scores = []
        for i, estimator in enumerate(model.estimators_):
            mse = mean_squared_error(Y[:, i], Y_pred[:, i])
            mse_scores.append(mse)

        weights = 1 / np.array(mse_scores)
        weights /= weights.sum() 
        # Extract feature importances per coordinate
        feature_importances = []
        for i, estimator in enumerate(model.estimators_):
            imp_df = pd.DataFrame({
                'Feature': adata.var.index,
                'Importance': estimator.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            feature_importances.append(imp_df)
        for k in range(n_dim):
            coord_dict_raw[keys[k]] = feature_importances[k]

        # Generate randomized matrix by permuting rows
        num_rows = X.shape[0]
        shuffled_indices = np.random.permutation(num_rows)
        random_matrix = X[shuffled_indices, :]
        lgb_model2 = lgb.LGBMRegressor(**params)
        model2 = MultiOutputRegressor(lgb_model2)
        model2.fit(random_matrix, Y)

        Y_pred2 = model2.predict(random_matrix)
        # Train model on randomized data
        weights = []
        mse_scores = []
        for i, estimator in enumerate(model2.estimators_):
            mse = mean_squared_error(Y[:, i], Y_pred[:, i])
            mse_scores.append(mse)
        
        weights = 1 / np.array(mse_scores)
        weights /= weights.sum()

        random_importances = np.zeros(adata.shape[1])
        for i, estimator in enumerate(model2.estimators_):
            random_importances += estimator.feature_importances_ * weights[i]
        
        for k in range(n_dim):
            coord_dict_rand[keys[k]] = model2.estimators_[k].feature_importances_
        
        # Compare real vs random importances to keep only significant genes
        if sps:
            coord_dict_raw = calculate_sps(coord_dict_raw, coord_dict_rand, n_dim)
        else:
            coord_dict_raw = keep_variant_genes(coord_dict_raw, coord_dict_rand, n_dim, threshold=threshold)
        return coord_dict_raw

def TVG_detection(
    adata: AnnData,
    coords: np.ndarray,
    sps: bool = False,
    threshold: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """
    Detect Temporally Variable Genes (TVGs) using regression on a 1D time coordinate.

    Args:
    adata : AnnData
        An AnnData object containing gene expression matrix (adata.X) and gene names (adata.var.index).
    coords : np.ndarray
        1D temporal coordinate of cells (shape: [n_cells, 1]).
    sps : bool, optional
        Whether to compute sps by comparing with random feature importances (default: False).

    Returns:
        coord_dict_raw : dict
        Dictionary containing important genes over time axis 'T',
        filtered by sps.
    """
    np.random.seed(0)
    n_dim = coords.shape[1]
    keys = ['T']
    X = adata.X
    Y = coords.copy()
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
    if n_dim != 1:
        raise ValueError("n_dim must be 1.")
    else:
        coord_dict_raw = {}
        coord_dict_rand = {}

        model = lgb_model
        model.fit(X, Y)
        Y_pred = model.predict(X)
        imp_df = pd.DataFrame({
                'Feature': adata.var.index,
                'Importance': model.booster_.feature_importance(importance_type='gain')
            }).sort_values(by='Importance', ascending=False)
        coord_dict_raw['T'] = imp_df

        # Generate randomized matrix by permuting rows
        #random_matrix = X.sample(frac=1).reset_index(drop=True)
        num_rows = X.shape[0]
        shuffled_indices = np.random.permutation(num_rows)
        random_matrix = X[shuffled_indices, :]
        lgb_model2 = lgb.LGBMRegressor(**params)
        model2 = lgb_model2
        model2.fit(random_matrix, Y)
        Y_pred2 = model2.predict(random_matrix)
        coord_dict_rand['T'] = model2.booster_.feature_importance(importance_type='gain')
        if sps:
            coord_dict_raw = calculate_sps(coord_dict_raw, coord_dict_rand, n_dim, keys=['T'])
        else:
            coord_dict_raw = keep_variant_genes(coord_dict_raw, coord_dict_rand, n_dim, threshold=threshold, keys=['T'])
        return coord_dict_raw

def gene_modules(adata, gene_list):
    df = adata[:, gene_list].to_df()
    corr = df.corr()
    Z = linkage(corr, 'complete', metric='correlation')
    return Z, corr, df