import math
import jax
import jax.numpy as jnp
import blackjax
from typing import Tuple, Callable, Any, Union

StepFnState = Any  # Tree
StepFnAndMomentState = Tuple[
    StepFnState,
    blackjax.adaptation.mass_matrix.WelfordAlgorithmState]
StepFnType = Callable[
    [
        jax.random.KeyArray,
        StepFnState,
    ],
    Tuple[
        jnp.ndarray,  # position or sample
        StepFnState,
    ],
]


def mcmc_diagonal(
    step_fn: StepFnType,
    step_fn_initial_state: StepFnState,
    prng_key: jax.random.KeyArray,
    dim: int,
    num_chains: int,
    num_samples: int,
    thinning: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    if num_samples % num_chains != 0:
        raise ValueError

    if thinning != 1:
        raise ValueError

    moment_init, moment_update, moment_final = (
        blackjax
        .adaptation
        .mass_matrix
        .welford_algorithm(
            is_diagonal_matrix=True))

    @jax.jit
    def one_step(
            state: StepFnAndMomentState,
            rng_key: jax.random.KeyArray,
    ) -> Tuple[StepFnAndMomentState, None]:
        step_fn_state, moment_state = state
        sample, step_fn_state = step_fn(
            rng_key,
            step_fn_state)
        moment_state = moment_update(
            moment_state,
            sample)
        return (step_fn_state, moment_state), None

    moment_state = moment_init(n_dims=dim)
    num_samples_per_chain = math.ceil(
        num_samples / num_chains)
    for _ in range(num_chains):
        # We will re-initialize the parameter, but reuse the
        # moment estimation from previous chain.
        initial_state = (
            step_fn_initial_state,
            moment_state)

        prng_key, subkey = jax.random.split(prng_key)
        scan_subkeys = jax.random.split(
            subkey,
            num_samples_per_chain)
        (_, moment_state), _ = jax.lax.scan(
            one_step,
            initial_state,
            scan_subkeys)

    Sigma, _, mu = moment_final(moment_state)
    return mu, Sigma
