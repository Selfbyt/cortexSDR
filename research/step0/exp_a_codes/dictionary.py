"""Dictionary learning for tile reconstruction.

Step 0 uses sklearn's MiniBatchDictionaryLearning — solid, well-tested,
fast enough for 1B-scale models. K-SVD from scratch is on the roadmap if
sklearn becomes a bottleneck or if we want binary-aware atom updates.

Dictionary convention: D shape is (K, C) where K=atom count, C=tile width.
A tile (1, C) is reconstructed as s @ D where s is (1, K) sparse.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning


def learn_dictionary(
    tile_matrix: np.ndarray,
    *,
    n_atoms: int,
    n_iter: int = 200,
    batch_size: int = 256,
    transform_n_nonzero_coefs: int = 16,
    random_state: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """Fit a dictionary on stacked tile rows.

    tile_matrix: (n_tiles, tile_dim)
    Returns: D shape (n_atoms, tile_dim), unit-norm rows.
    """
    if tile_matrix.shape[0] < n_atoms:
        raise ValueError(
            f"Too few tiles ({tile_matrix.shape[0]}) to fit {n_atoms} atoms"
        )
    learner = MiniBatchDictionaryLearning(
        n_components=n_atoms,
        max_iter=n_iter,
        batch_size=batch_size,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=transform_n_nonzero_coefs,
        random_state=random_state,
        verbose=2 if verbose else 0,
        n_jobs=1,
    )
    learner.fit(tile_matrix.astype(np.float32))
    D = learner.components_.astype(np.float32)  # (n_atoms, tile_dim), already unit-norm
    return D


def random_atom_init(n_atoms: int, tile_dim: int, *, seed: int = 0) -> np.ndarray:
    """Random unit-norm atoms. Useful for sanity comparisons."""
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n_atoms, tile_dim)).astype(np.float32)
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    return D / np.maximum(norms, 1e-9)
