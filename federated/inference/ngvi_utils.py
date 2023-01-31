import jax
import jax.numpy as jnp
import chex
import optax
import functools
from optax._src import base
from optax._src import utils
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import alias
from tensorflow_probability.substrates import jax as tfp
from typing import Tuple, Optional, Any, Type, NamedTuple

from ..utils import config_types
from ..modules.utils import (
    ModelIndex,
)
from .optimization_utils import clip_by_global_mean_and_std
from ..objectives.gaussians import DiagonalGaussian
from ..objectives.logistics_regression import (
    Dataset,
    SimpleObjective,
    ObjectiveWithDiagGaussianPrior,
)

if optax.__version__ != "0.1.3":
    raise ValueError("`optax` needs to be 0.1.3 for the Adam modifications")


class ScaleByVIAdamState(NamedTuple):
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    Fisher: optax.Params
    Lambda: optax.Params


def scale_by_viadam(
    num_points: int,
    prior_Lambda: optax.Params,
    initial_Lambda_scale: float,
    b1: float = 0.9,
    b2: float = 0.999,
    damping: float = 0.0,
    apply_bias_correction: bool = False,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def Lambda_fn(nu: base.Updates) -> optax.Params:
        return jax.tree_util.tree_map(
            lambda v, pl: num_points * v + pl,
            nu, prior_Lambda)

    def init_fn(params: optax.Params) -> ScaleByVIAdamState:
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(  # Second moment
            lambda t: jnp.ones_like(t) * initial_Lambda_scale, params)
        return ScaleByVIAdamState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            nu=nu,
            Fisher=jnp.zeros_like(params),
            Lambda=Lambda_fn(nu))

    def update_fn(
        updates: optax.Params,
        state: ScaleByVIAdamState,
        params: Optional[optax.Params] = None,
    ) -> Tuple[optax.Params, ScaleByVIAdamState]:
        del params
        mu = transform.update_moment(updates, state.mu, b1, 1)
        # We use first moment of the "Fisher", which is the second moment.
        nu = transform.update_moment_per_elem_norm(state.Fisher, state.nu, b2, 1)
        count_inc = numerics.safe_int32_increment(state.count)

        if apply_bias_correction is True:
            mu_hat = transform.bias_correction(mu, b1, count_inc)
            nu_hat = transform.bias_correction(nu, b2, count_inc)
        else:
            mu_hat = mu
            nu_hat = nu

        updates = jax.tree_util.tree_map(
            lambda m, v, pl: m / (v + pl / num_points + damping),
            mu_hat, nu_hat, prior_Lambda)
        mu = utils.cast_tree(mu, mu_dtype)
        return updates, ScaleByVIAdamState(
            count=count_inc,
            mu=mu,
            nu=nu,
            Fisher=jnp.zeros_like(state.Fisher),
            Lambda=Lambda_fn(nu))

    return base.GradientTransformation(init_fn, update_fn)


def viadam_state_replace_Fisher(
    Fisher: optax.Params,
    state: optax.OptState,
) -> optax.OptState:
    new_state = []
    for s in state:
        if isinstance(s, ScaleByVIAdamState):
            new_s = ScaleByVIAdamState(
                count=s.count,
                mu=s.mu,
                nu=s.nu,
                Fisher=Fisher,
                Lambda=s.Lambda)
            new_state.append(new_s)
        else:
            new_state.append(s)

    return tuple(new_state)


def viadam(
    learning_rate: alias.ScalarOrSchedule,
    num_points: int,
    prior_Lambda: optax.Params,
    initial_Lambda_scale: float,
    b1: float = 0.9,
    b2: float = 0.999,
    damping: float = 0.0,
    apply_bias_correction: bool = False,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_viadam(
            num_points=num_points,
            prior_Lambda=prior_Lambda,
            initial_Lambda_scale=initial_Lambda_scale,
            b1=b1,
            b2=b2,
            damping=damping,
            apply_bias_correction=apply_bias_correction,
            mu_dtype=mu_dtype),
        alias._scale_by_learning_rate(learning_rate),
    )


def compute_grads_and_Fisher_from_sample(
    q_mu: optax.Params,
    q_Sigma: optax.Params,
    objective_type: Type[SimpleObjective],
    model_index: ModelIndex,
    prng_key: jax.random.KeyArray,
    data_batch: Dataset,
    num_classes: int,
    num_points: int,
    **objective_kwargs,
) -> Tuple[optax.Params, optax.Params]:
    if objective_type != ObjectiveWithDiagGaussianPrior:
        raise TypeError("This only works for diagonal objective.")
    subkey_s, subkey_g, subkey_F = jax.random.split(prng_key, num=3)

    params_sample = DiagonalGaussian.compute_sample(
        prng_key=subkey_s,
        mu=q_mu,
        Sigma=q_Sigma)

    grads = objective_type.compute_grad_for_train(
        model_index=model_index,
        params=params_sample,
        prng_key=subkey_g,
        data_batch=data_batch,
        num_classes=num_classes,
        num_points=num_points,
        **objective_kwargs)

    Fisher = objective_type.compute_Fisher_for_train(
        model_index=model_index,
        params=params_sample,
        prng_key=subkey_F,
        data_batch=data_batch,
        num_classes=num_classes,
        num_points=num_points,
        reduce_op="MEAN")

    return grads, Fisher


def get_viadam_state(
    state: optax.OptState,
) -> ScaleByVIAdamState:
    _states = []
    for s in state:
        if isinstance(s, ScaleByVIAdamState):
            _states.append(s)

    if len(_states) > 1:
        raise ValueError

    return _states[0]


def ngvi_diagonal(
    config: config_types.MomentConfig,
    objective_type: Type[SimpleObjective],
    model_index: ModelIndex,
    data: Dataset,
    num_classes: int,
    num_points: int,
    batch_size: int,
    prng_key: jax.random.KeyArray,
    initial_params: optax.Params,
    **objective_kwargs,
) -> Tuple[optax.Params, optax.Params]:

    optimizer = viadam(
        learning_rate=config.mu_learning_rate,
        num_points=num_points,
        prior_Lambda=objective_kwargs["prior_diag_Lambda"],
        initial_Lambda_scale=config.initial_Lambda_scale,
        b2=config.Lambda_decay_rate)

    def step_fn(
        carry: Tuple[optax.Params, optax.OptState],
        inputs: Tuple[jax.random.KeyArray, Dataset],
    ) -> Tuple[Tuple[optax.Params, optax.OptState], None]:
        params, opt_state = carry
        prng_key, data_batch = inputs

        def _fn(subkey: jax.random.KeyArray) -> Tuple[optax.Params, optax.Params]:
            q_Sigma = 1. / get_viadam_state(opt_state).Lambda
            return compute_grads_and_Fisher_from_sample(
                q_mu=params,
                q_Sigma=q_Sigma,
                objective_type=objective_type,
                model_index=model_index,
                prng_key=subkey,
                data_batch=data_batch,
                num_classes=num_classes,
                num_points=num_points,
                **objective_kwargs)

        subkeys = jax.random.split(prng_key, config.num_samples)
        grads_samples, Fisher_samples = jax.vmap(_fn)(subkeys)
        grads_samples_mean = jnp.mean(grads_samples, axis=0)
        Fisher_samples_mean = jnp.mean(Fisher_samples, axis=0)

        if config.clip_ratio is not None:
            Fisher_samples_mean = clip_by_global_mean_and_std(
                Fisher_samples_mean, num=config.clip_ratio)

        opt_state = viadam_state_replace_Fisher(
            Fisher=Fisher_samples_mean,
            state=opt_state)
        updates, opt_state = optimizer.update(
            grads_samples_mean,
            opt_state,
            params)
        params = optax.apply_updates(
            params,
            updates)

        return (params, opt_state), None

    def init_fn(params: optax.Params) -> Tuple[optax.Params, optax.OptState]:
        return params, optimizer.init(params)

    indices = jnp.arange(
        config.num_epochs *
        num_points)

    if indices.shape[0] > data.Xs.shape[0]:
        raise ValueError(
            f"Not enough data ({data.Xs.shape[0]}) for "
            f"{config.num_epochs} epochs, which requires "
            f"{config.num_epochs} x {num_points} "
            f"= {indices.shape[0]} points.")

    data_subset = jax.tree_util.tree_map(
        lambda A: jnp.take(A, indices, axis=0),
        data)

    prng_key, subkey = jax.random.split(prng_key)
    data_batches = objective_type.generate_data_batches(
        prng_key=subkey,
        data=data_subset,
        batch_size=batch_size)

    num_batches = data_batches.Xs.shape[0]
    scan_subkeys = jax.random.split(
        prng_key,
        num_batches)

    (final_mu, final_opt_state), unused_outputs = jax.lax.scan(
        step_fn,
        init_fn(initial_params),
        (scan_subkeys, data_batches))

    final_Lambda = scale_Lambda(
        scale=config.scale,
        Lambda=get_viadam_state(final_opt_state).Lambda,
        weight_decay=config.weight_decay * num_points,
        prior_Lambda=objective_kwargs["prior_diag_Lambda"])
    final_Sigma = (1. / final_Lambda)
    return final_mu, final_Sigma


def scale_Lambda(
    scale: float,
    Lambda: optax.Params,
    weight_decay: float,
    prior_Lambda: optax.Params,
) -> optax.Params:
    # scaled Lambda_{\k}
    # = c Lambda_k + Lambda_{-k}
    # = c [ Lambda_{\k} - Lambda_{-k} ] + Lambda_{-k}
    # = c [ Lambda_{\k} + wd - wd - Lambda_{-k} ] + wd - wd + Lambda_{-k}
    # = c [ Lambda_{\k} + wd - Lambda_{prior} ] - wd + Lambda_{prior}
    # = c [ Lambda_{\k} - [ Lambda_{prior} - wd ] ] + [ Lambda_{prior} - wd ]
    # Note that `prior_diag_Lambda` here already includes the weight decay term
    # so if we scale this, we need to subtract one from the scale.
    def _fn(L: jnp.ndarray, pL: jnp.ndarray) -> jnp.ndarray:
        cL = pL - weight_decay  # cavity
        lL = L - cL  # local
        return scale * lL + cL

    return jax.tree_util.tree_map(_fn, Lambda, prior_Lambda)


def scaled_identity(
    config: config_types.MomentConfig,
    num_points: int,
    initial_params: optax.Params,
    **objective_kwargs,
) -> Tuple[optax.Params, optax.Params]:
    unscaled_Lambda = (
        num_points *
        config.initial_Lambda_scale *
        jnp.ones_like(initial_params) +
        objective_kwargs["prior_diag_Lambda"])
    final_Lambda = scale_Lambda(
        scale=config.scale,
        Lambda=unscaled_Lambda,
        weight_decay=config.weight_decay * num_points,
        prior_Lambda=objective_kwargs["prior_diag_Lambda"])
    return final_Lambda, 1. / final_Lambda


def laplace_diagonal(
    config: config_types.MomentConfig,
    objective_type: Type[SimpleObjective],
    model_index: ModelIndex,
    data: Dataset,
    num_classes: int,
    num_points: int,
    batch_size: int,
    prng_key: jax.random.KeyArray,
    initial_params: optax.Params,
    **objective_kwargs,
) -> Tuple[optax.Params, optax.Params]:
    if objective_type != ObjectiveWithDiagGaussianPrior:
        raise TypeError("This only works for diagonal objective.")

    def step_fn(
        carry: optax.Params,
        inputs: Tuple[jax.random.KeyArray, Dataset],
    ) -> Tuple[optax.Params, None]:
        subkey, data_batch = inputs
        Fisher = objective_type.compute_Fisher_for_train(
            model_index=model_index,
            params=initial_params,
            prng_key=subkey,
            data_batch=data_batch,
            num_classes=num_classes,
            num_points=num_points,
            reduce_op="MEAN")
        return carry + Fisher, None

    indices = jnp.arange(
        config.num_epochs *
        num_points)

    if indices.shape[0] > data.Xs.shape[0]:
        raise ValueError(
            f"Not enough data ({data.Xs.shape[0]}) for "
            f"{config.num_epochs} epochs, which requires "
            f"{config.num_epochs} x {num_points} "
            f"= {indices.shape[0]} points.")

    data_subset = jax.tree_util.tree_map(
        lambda A: jnp.take(A, indices, axis=0),
        data)

    prng_key, subkey = jax.random.split(prng_key)
    data_batches = objective_type.generate_data_batches(
        prng_key=subkey,
        data=data_subset,
        batch_size=min(batch_size, num_points))

    num_batches = data_batches.Xs.shape[0]
    scan_subkeys = jax.random.split(
        prng_key,
        num_batches)

    Fisher, unused_outputs = jax.lax.scan(
        step_fn,
        jnp.zeros_like(initial_params),
        (scan_subkeys, data_batches))

    # We want the Fisher to represent that of a dataset.
    Fisher = Fisher * num_points / num_batches

    # Clipping before adding the prior Lambda.
    if config.clip_ratio is not None:
        Fisher = clip_by_global_mean_and_std(
            Fisher, num=config.clip_ratio)

    Lambda = Fisher + objective_kwargs["prior_diag_Lambda"]
    final_Lambda = scale_Lambda(
        scale=config.scale,
        Lambda=Lambda,
        weight_decay=config.weight_decay * num_points,
        prior_Lambda=objective_kwargs["prior_diag_Lambda"])
    return final_Lambda, 1. / final_Lambda
