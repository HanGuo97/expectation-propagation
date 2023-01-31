import math
import jax
import jax.numpy as jnp
import optax
import blackjax
import functools
from typing import Tuple, Dict, Optional, Callable, Type

from .optimization_utils import (
    Dataset,
    simple_optimization,
    create_optimization_functions)
from ..objectives.gaussians import (
    Gaussian,
    DiagonalGaussian)
from ..objectives.logistics_regression import SimpleObjective
from . import sampling_utils
from . import ngvi_utils
from . import moments_utils
from ..utils import config_types
from ..modules.utils import ModelIndex


@functools.partial(
    jax.jit,
    static_argnames=(
        "sample_config",
        "moment_config",
        "objective_type",
        "model_index",
        "dim",
        # "data",
        "num_classes",
        "num_epochs",
        "num_points",
        "batch_size",
        # "prng_key",
        # "init_theta",
    ),
)
def mcmc_laplace_diagonal(
    sample_config: config_types.SampleOptimizationConfig,
    moment_config: config_types.MomentConfig,
    objective_type: Type[SimpleObjective],
    model_index: ModelIndex,
    dim: int,
    data: Dataset,
    num_classes: int,
    num_epochs: int,
    num_points: int,
    batch_size: int,
    prng_key: jax.random.KeyArray,
    init_theta: jnp.ndarray,
    **objective_kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    if math.ceil(
        sample_config.num_samples /
        sample_config.num_chains
    ) > num_epochs:
        raise ValueError(
            f"The dataset has {num_epochs} epochs. However, we are "
            f"trying to generate {sample_config.num_samples} with "
            f"{sample_config.num_chains} chains.")

    sample_init_fn, sample_epoch_fn = create_optimization_functions(
        config=sample_config.optim_config,
        objective_type=objective_type,
        model_index=model_index,
        num_classes=num_classes,
        num_points=num_points,
        batch_size=batch_size,
        **objective_kwargs)

    data_epochs = objective_type.generate_data_epochs(
        data=data,
        num_epochs=num_epochs,
        num_points=num_points)

    def mcmc_step_fn(
        rng_key: jax.random.KeyArray,
        sgd_state: Tuple[int, Tuple[optax.Params, optax.OptState]],
    ) -> Tuple[jnp.ndarray, Tuple[int, Tuple[optax.Params, optax.OptState]]]:
        epoch_index, carry = sgd_state
        data_epoch = jax.tree_util.tree_map(
            lambda A: jnp.take(A, epoch_index, axis=0),
            data_epochs)
        inputs = (rng_key, data_epoch)
        carry, _ = sample_epoch_fn(carry, inputs)
        return carry[0], (epoch_index + 1, carry)

    prng_key, subkey = jax.random.split(prng_key)
    mu, _Sigma = sampling_utils.mcmc_diagonal(
        step_fn=mcmc_step_fn,
        step_fn_initial_state=(0, sample_init_fn(init_theta)),
        prng_key=subkey,
        dim=dim,
        num_chains=sample_config.num_chains,
        num_samples=sample_config.num_samples,
        thinning=sample_config.thinning)

    if moment_config.method == "none":
        Sigma = jnp.zeros_like(mu)
    if moment_config.method == "identity":
        _, Sigma = ngvi_utils.scaled_identity(
            config=moment_config,
            num_points=num_points,
            initial_params=mu,
            **objective_kwargs)
    if moment_config.method == "laplace":
        prng_key, subkey = jax.random.split(prng_key)
        _, Sigma = ngvi_utils.laplace_diagonal(
            config=moment_config,
            objective_type=objective_type,
            model_index=model_index,
            data=data,
            num_classes=num_classes,
            num_points=num_points,
            batch_size=batch_size,
            prng_key=subkey,
            initial_params=mu,
            **objective_kwargs)
    if moment_config.method == "vi":
        prng_key, subkey = jax.random.split(prng_key)
        _, Sigma = ngvi_utils.ngvi_diagonal(
            config=moment_config,
            objective_type=objective_type,
            model_index=model_index,
            data=data,
            num_classes=num_classes,
            num_points=num_points,
            batch_size=batch_size,
            prng_key=subkey,
            initial_params=mu,
            **objective_kwargs)
    if moment_config.method == "mcmc":
        _, Sigma = moments_utils.diagonal_covariance_postprocess(
            config=moment_config,
            num_points=num_points,
            initial_Sigma=_Sigma,
            **objective_kwargs)

    return mu, Sigma


@functools.partial(
    jax.jit,
    static_argnames=(
        "sample_config",
        "moment_config",
        "objective_type",
        "model_index",
        "dim",
        # "data",
        "num_classes",
        "num_epochs",
        "num_points",
        "batch_size",
        # "prng_key",
        # "init_theta",
    ),
)
def map_laplace_diagonal(
    sample_config: config_types.SampleOptimizationConfig,
    moment_config: Optional[config_types.MomentConfig],
    objective_type: Type[SimpleObjective],
    model_index: ModelIndex,
    dim: int,
    data: Dataset,
    num_classes: int,
    num_epochs: int,
    num_points: int,
    batch_size: int,
    prng_key: jax.random.KeyArray,
    init_theta: jnp.ndarray,
    **objective_kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    if sample_config.num_samples > num_epochs:
        raise ValueError(
            f"The dataset has {num_epochs} epochs. However, we are "
            f"trying to generate {sample_config.num_samples}.")

    prng_key, subkey = jax.random.split(prng_key)
    mu = simple_optimization(
        data=data,
        prng_key=subkey,
        initial_params=init_theta,
        config=sample_config,
        objective_type=objective_type,
        model_index=model_index,
        num_classes=num_classes,
        num_points=num_points,
        batch_size=batch_size,
        **objective_kwargs)

    if moment_config is None:
        Sigma = jnp.zeros_like(mu)
    else:
        prng_key, subkey = jax.random.split(prng_key)
        Sigma, _ = laplace_approximation_diagonal(
            data=data,
            prng_key=subkey,
            params=mu,
            config=moment_config,
            objective_type=objective_type,
            model_index=model_index,
            num_classes=num_classes,
            num_points=num_points,
            **objective_kwargs)

    return mu, Sigma


def optimize(
    prng_key: jax.random.KeyArray,
    objective: SimpleObjective,
    init_theta: jnp.ndarray,
    config: config_types.SampleOptimizationConfig,
    debug_callback: Optional[Callable] = None,
) -> jnp.ndarray:
    return simple_optimization(
        data=objective.data,
        prng_key=prng_key,
        initial_params=init_theta,
        config=config,
        objective_type=type(objective),
        model_index=objective.model_index,
        num_classes=objective.num_classes,
        num_points=objective.num_points,
        batch_size=objective.batch_size,
        debug_callback=debug_callback,
        **objective.kwargs,
    )


def approximate_inference(
    method: str,
    prng_key: jax.random.KeyArray,
    objective: SimpleObjective,
    init_theta: jnp.ndarray,
    sample_config: config_types.SampleOptimizationConfig,
    moment_config: config_types.MomentConfig,
    shared_basis_Q: Optional[jnp.ndarray] = None,
) -> Gaussian:

    if objective.kwargs["prior_strength"] != 1.0:
        raise ValueError(
            "We hard-coded this in the following"
            "functions to simplify the codes.")

    if method == "mcmc-diagonal":
        mu, Sigma = mcmc_laplace_diagonal(
            sample_config=sample_config,
            moment_config=moment_config,
            objective_type=type(objective),
            model_index=objective.model_index,
            dim=objective.dim,
            data=objective.data,
            num_classes=objective.num_classes,
            num_epochs=objective.num_epochs,
            num_points=objective.num_points,
            batch_size=objective.batch_size,
            prng_key=prng_key,
            init_theta=init_theta,
            **objective.kwargs)

        return DiagonalGaussian(
            mu=mu,
            Sigma=Sigma)

    if method == "map-diagonal":
        mu, Sigma = map_laplace_diagonal(
            sample_config=sample_config,
            moment_config=moment_config,
            objective_type=type(objective),
            model_index=objective.model_index,
            dim=objective.dim,
            data=objective.data,
            num_classes=objective.num_classes,
            num_epochs=objective.num_epochs,
            num_points=objective.num_points,
            batch_size=objective.batch_size,
            prng_key=prng_key,
            init_theta=init_theta,
            **objective.kwargs)

        return DiagonalGaussian(
            mu=mu,
            Sigma=Sigma)

    raise ValueError(f"Unknown sampler name: {method}")
