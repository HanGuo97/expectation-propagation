import jax
import jax.numpy as jnp
from typing import Tuple

from ..utils import config_types
from .ngvi_utils import scaled_identity
from .optimization_utils import clip_by_global_mean_and_std


def shrunk_covariance_diagonal(
    Sigma: jnp.ndarray,
    shrinkage: float,
) -> jnp.ndarray:
    Sigma_prior = jnp.ones_like(Sigma)
    shrunk_Sigma = (
        shrinkage * Sigma_prior +
        (1. - shrinkage) * Sigma)
    return shrunk_Sigma


def diagonal_covariance_postprocess(
    config: config_types.MomentConfig,
    num_points: int,
    initial_Sigma: jnp.ndarray,
    **objective_kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    Sigma = shrunk_covariance_diagonal(
        Sigma=initial_Sigma,
        shrinkage=config.shrinkage)

    # We will operate on Lambda
    Lambda = 1. / Sigma
    if config.clip_ratio is not None:
        Lambda = clip_by_global_mean_and_std(
            Lambda, num=config.clip_ratio)

    reference_Lambda, _ = scaled_identity(
        config=config,
        num_points=num_points,
        # only the dimension, dtype, and device matter
        initial_params=initial_Sigma,
        **objective_kwargs)

    scale = (
        jnp.linalg.norm(reference_Lambda) /
        jnp.linalg.norm(Lambda))

    final_Lambda = Lambda * scale
    return final_Lambda, 1. / final_Lambda
