# coding=utf-8
# Copyright 2020 Maruan Al-Shedivat.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Federated learning algorithms."""

from typing import Any, Callable, Dict, List, Tuple

import attr
import wandb
import optax
import jax.numpy as jnp
from jax import random
from tqdm.auto import trange

from ..inference import utils as inference_utils
from ..inference import optimization_utils
from ..experiments import evaluation_utils
from ..experiments import data_utils
from ..objectives import gaussians
from ..objectives.base import StochasticObjective
from ..utils.timing import Timer
from ..utils.types import (
    ServerState,
    RoundInfo,
    ClientUpdateFn,
    ServerUpdateFn,
    SampleClientsFn,
    FederatedLearningFn)
from .algorithms import (
    compute_weighted_average,
    sample_clients_uniformly,
)
from ..utils import config_types
from ..modules.utils import ModelIndex
from ..experiments import gaussian_utils

SAVE_FREQUENCY = 100


def fed_opt(
    data_helper: data_utils.DataHelper,
    client_update_fn: ClientUpdateFn,
    server_update_fn: ServerUpdateFn,
    sample_clients_fn: SampleClientsFn,
    server_optimizer: optax.GradientTransformation,
    prng_key: jnp.ndarray,
    init_state: jnp.ndarray,
    num_rounds: int,
    num_clients_per_round: int,
) -> Tuple[List[ServerState], List[RoundInfo]]:
    """Runs generalized federated averaging for the specified number of rounds.

    At each round, the algorithm does the following:
        1.  Samples a batch of clients using `sample_clients_fn`.
        2.  Runs `client_update_fn` on each sampled client objective that
            returns a `client_delta`.
        3.  Aggregates `client_deltas` using `server_update_fn`.

    Args:
        client_update_fn: A function for computing local client updates.
        server_update_fn: A function for computing server updates.
        sample_clients_fn: A function for sampling indices of the clients.
        client_objectives: A list of client objective functions.
        prng_key: A key for random number generation.
        init_state: The initial server state.
        num_rounds: The number of training rounds to run.
        num_clients_per_round: The number of clients used at each round.

    Returns:
        A list of tuples `(round: int, state: ServerState)` that represents the
        trajectory of the server state over the course of training.
    """

    server_state = ServerState(
        r=0,
        x=init_state,
        v=server_optimizer.init(init_state))

    trajectory = [server_state]
    info = [None]
    for round_index in trange(num_rounds):
        round_info = {}

        # Select clients.
        prng_key, subkey = random.split(prng_key)
        with Timer("select_clients_time") as t:
            client_ids = sample_clients_fn(
                subkey, data_helper.num_train_clients, num_clients_per_round
            )
            client_objectives_round = [
                data_helper.get_client_train_objective(i)
                for i in client_ids]
            client_weights_round = jnp.asarray(
                [float(o.num_points) for o in client_objectives_round]
            )
        round_info[t.description] = t.elapsed

        # Compute client updates.
        client_deltas_round = []
        # TODO: parallelize this loop.
        with Timer("client_updates_time") as t:
            for client_objective in client_objectives_round:
                prng_key, subkey = random.split(prng_key)
                client_delta = client_update_fn(
                    client_objective, server_state.x, subkey
                )
                client_deltas_round.append(client_delta)
        round_info[t.description] = t.elapsed

        # Update server state.
        with Timer("server_update_time") as t:
            server_state = server_update_fn(
                client_deltas_round, client_weights_round, server_state
            )
        round_info[t.description] = t.elapsed
        # trajectory.append(server_state)
        info.append(round_info)

        server_dist = gaussians.DiagonalGaussian(
            mu=server_state.x,
            Sigma=jnp.ones_like(server_state.x),
        )
        wandb.log({
            "round_index": round_index,
            **evaluation_utils.evaluate_objectives(
                dist=server_dist,
                objective_train=data_helper.centralized_train_objective,
                objective_eval=data_helper.centralized_test_objective),
            },
        )
        if round_index % SAVE_FREQUENCY == 0:
            trajectory.append(server_state)

    return trajectory, info


def create_fed_avg(
    *,
    server_optim_config: config_types.OptimizerConfig,
    sample_config: config_types.SampleOptimizationConfig,
    weight_decay: float = 0.0,
) -> FederatedLearningFn:
    """Creates a generalized FedAvg.

    Args:
        client_steps_per_round: The number of local SGD steps done by clients.
        client_learning_rate_schedule: The schedule for client learning rate.
        server_learning_rate_schedule: The schedule for server learning rate.
        client_momentum: The momentum used by client optimizers.
        server_momentum: The momentum used by the server optimizer.

    Returns:
        A federated learning function.
    """

    server_optimizer = optimization_utils.create_optimizer(
        name=server_optim_config.name,
        learning_rate=server_optim_config.learning_rate,
        max_norm=server_optim_config.max_norm,
        **server_optim_config.kwargs)

    def _client_update_fn(
        objective: StochasticObjective,
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
    ) -> jnp.ndarray:

        cavity_dist = gaussian_utils.get_uniform_gaussian(
            "diagonal",
            d=objective.dim,
            Q=None)

        tilted_prior = gaussian_utils.add_weight_decay(
            "diagonal",
            q=cavity_dist,
            # The weight decay value is set for average loss
            # but we use sum loss here, so scale it accordingly.
            c=weight_decay * objective.num_points,
            d=objective.dim,
            Q=None)

        # tilted distributions
        tilted_objective = gaussian_utils.create_tilted_objective(
            gtype="diagonal",
            objective=objective,
            prior=tilted_prior,
            prior_strength=1.0)

        tilted_dist = inference_utils.approximate_inference(
            method="map-diagonal",
            prng_key=prng_key,
            objective=tilted_objective,
            init_theta=init_state,
            sample_config=sample_config,
            moment_config=None,
            shared_basis_Q=None)
        x = tilted_dist.mu

        return init_state - x

    def _server_update_fn(
        client_deltas: List[jnp.ndarray],
        client_weights: jnp.ndarray,
        init_state: ServerState,
    ) -> ServerState:
        return compute_server_update(
            client_deltas=client_deltas,
            client_weights=client_weights,
            init_state=init_state,
            server_optimizer=server_optimizer,
        )

    def _fed_learn(
        data_helper: data_utils.DataHelper,
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
        num_rounds: int,
        num_clients_per_round: int,
    ) -> Tuple[List[ServerState], List[RoundInfo]]:
        return fed_opt(
            data_helper=data_helper,
            client_update_fn=_client_update_fn,
            server_update_fn=_server_update_fn,
            sample_clients_fn=sample_clients_uniformly,
            server_optimizer=server_optimizer,
            prng_key=prng_key,
            init_state=init_state,
            num_rounds=num_rounds,
            num_clients_per_round=num_clients_per_round,
        )

    return _fed_learn


def compute_server_update(
    client_deltas: List[jnp.ndarray],
    client_weights: jnp.ndarray,
    init_state: ServerState,
    server_optimizer: optax.GradientTransformation,
) -> ServerState:
    # Compute the weighted average of client deltas.
    client_deltas_avg = compute_weighted_average(client_deltas, client_weights)
    # Updates
    updates, v = server_optimizer.update(client_deltas_avg, init_state.v)
    x = optax.apply_updates(init_state.x, updates)
    return ServerState(r=(init_state.r + 1), x=x, v=v)
