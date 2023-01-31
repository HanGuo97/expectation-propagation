import jax
import chex
import optax
import functools
import jax.numpy as jnp
from typing import Callable, Tuple, Optional, Type
from optax._src.alias import ScalarOrSchedule
from optax._src import linear_algebra as optax_linalg

from ..utils import config_types
from ..modules.utils import (
    ModelIndex,
    apply_for_train,
)
from ..objectives.logistics_regression import (
    Dataset,
    SimpleObjective,
    ObjectiveWithDiagGaussianPrior,
)


def create_optimizer(
        name: str,
        learning_rate: ScalarOrSchedule,
        max_norm: Optional[float] = None,
        **kwargs) -> optax.GradientTransformation:
    if name == "adam":
        optimizer = optax.adam(learning_rate=learning_rate, **kwargs)
    if name == "sgd":
        optimizer = optax.sgd(learning_rate=learning_rate, **kwargs)
    if name == "adagrad":
        optimizer = optax.adagrad(learning_rate=learning_rate, **kwargs)
    if name == "sgld":
        raise ValueError("Optax has updated the implementation upstream.")
        # SGLD     : update = lr x g_t + sqrt(2 x lr) x scale x epsilon
        # Noisy SGD: update = lr x g_t + sqrt(eta x (1+t)^-gamma) x epislon
        #
        # where (in SGLD):
        # - epsilon is a unit Gaussian noise
        # - scale = 0.0 recovers the standard SGLD.
        #
        # Notice that in Noisy SGD, the noise part is not scaled by learning
        # rate. This is how `optax` implements it (oddly), as of April 26, 2022.
        #
        # Hence, in Noisy SGD, we have SGLD if we let
        # - gamma = 0.0
        # - eta = 2 x lr x scale^2
        if callable(learning_rate):
            raise NotImplementedError

        noise_eta = (
            2 *
            learning_rate *
            kwargs.get("noise_scale", 0.) ** 2
        )
        optimizer = optax.noisy_sgd(
            learning_rate=learning_rate,
            eta=noise_eta,
            gamma=0.,
        )

    if max_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_norm),
            optimizer,
        )

    return optimizer


def clip_by_global_norm(params: optax.Params, max_norm: float) -> optax.Params:
    g_norm = optax_linalg.global_norm(params)
    trigger = jnp.squeeze(g_norm < max_norm)
    chex.assert_shape(trigger, ())  # A scalar.

    params = jax.tree_util.tree_map(
        lambda t: jax.lax.select(
            trigger,
            t,
            (t / g_norm) * max_norm),
        params)
    return params


def clip_by_global_mean_and_std(params: jnp.ndarray, num: float = 2) -> jnp.ndarray:
    mean = params.mean()
    std = params.std()
    clip_min = mean - num * std
    clip_max = mean + num * std
    return jnp.clip(params, a_min=clip_min, a_max=clip_max)


@functools.partial(
    jax.jit,
    static_argnames=(
        "config",
        "objective_type",
        "model_index",
        "num_classes",
        "num_points",
        "batch_size",
        "debug_callback",
    ),
)
def simple_optimization(
    data: Dataset,
    prng_key: jax.random.KeyArray,
    initial_params: optax.Params,
    config: config_types.SampleOptimizationConfig,
    objective_type: Type[SimpleObjective],
    model_index: ModelIndex,
    num_classes: int,
    num_points: int,
    batch_size: int,
    debug_callback: Optional[Callable] = None,
    **objective_kwargs,
) -> optax.Params:

    init_fn, step_fn = create_optimization_functions(
        config=config.optim_config,
        objective_type=objective_type,
        model_index=model_index,
        num_classes=num_classes,
        num_points=num_points,
        batch_size=batch_size,
        debug_callback=debug_callback,
        **objective_kwargs)

    indices = jnp.arange(
        config.num_samples *
        num_points)

    if indices.shape[0] > data.Xs.shape[0]:
        raise ValueError(
            f"Not enough data ({data.Xs.shape[0]}) for "
            f"{config.num_samples} epochs, which requires "
            f"{config.num_samples} x {num_points} "
            f"= {indices.shape[0]} points.")

    data_subset = jax.tree_util.tree_map(
        lambda A: jnp.take(A, indices, axis=0),
        data)

    (final_params, _), _ = step_fn(
        init_fn(initial_params),
        (prng_key, data_subset))

    return final_params


def create_optimization_functions(
    config: config_types.OptimizerConfig,
    objective_type: Type[SimpleObjective],
    model_index: ModelIndex,
    num_classes: int,
    num_points: int,
    batch_size: int,
    debug_callback: Optional[Callable] = None,
    **objective_kwargs,
) -> Tuple[
    Callable[
        [optax.Params],
        Tuple[optax.Params, optax.OptState]
    ],
    Callable[
        [
            Tuple[optax.Params, optax.OptState],
            Tuple[jax.random.KeyArray, Dataset],
        ],
        Tuple[optax.Params, optax.OptState]
    ],
]:

    optimizer = create_optimizer(
        name=config.name,
        learning_rate=config.learning_rate,
        max_norm=config.max_norm,
        **config.kwargs)

    def one_step(
        carry: Tuple[optax.Params, optax.OptState],
        inputs: Tuple[jax.random.KeyArray, Dataset],
    ) -> Tuple[Tuple[optax.Params, optax.OptState], None]:
        params, opt_state = carry
        prng_key, data_batch = inputs
        grads = objective_type.compute_grad_for_train(
            model_index=model_index,
            params=params,
            prng_key=prng_key,
            data_batch=data_batch,
            num_classes=num_classes,
            num_points=num_points,
            **objective_kwargs)
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params)
        params = optax.apply_updates(
            params,
            updates)

        if debug_callback is not None:
            jax.experimental.host_callback.id_tap(
                debug_callback,
                {
                    "params": params,
                })

        return (params, opt_state), None

    def init(params: optax.Params) -> Tuple[optax.Params, optax.OptState]:
        return params, optimizer.init(params)

    def one_epoch(
        carry: Tuple[optax.Params, optax.OptState],
        inputs: Tuple[jax.random.KeyArray, Dataset],
    ) -> Tuple[Tuple[optax.Params, optax.OptState], None]:
        prng_key, data_epoch = inputs
        prng_key, subkey = jax.random.split(prng_key)
        data_batches = objective_type.generate_data_batches(
            prng_key=subkey,
            data=data_epoch,
            batch_size=batch_size)

        num_batches = data_batches.Xs.shape[0]
        scan_subkeys = jax.random.split(
            prng_key,
            num_batches)

        return jax.lax.scan(
            one_step,
            carry,
            (scan_subkeys, data_batches))

    return init, one_epoch
