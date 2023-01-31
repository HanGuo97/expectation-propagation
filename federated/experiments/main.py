import jax
import wandb
import click
import contexttimer
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from tqdm import trange
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple, Union, Dict, Callable, Optional, Any

from . import misc_utils
from . import data_utils
from . import file_utils
from . import gaussian_utils
from . import evaluation_utils
from ..utils import config_types
from ..inference import utils as inference_utils
from ..learning import algorithms
from ..learning import algorithms_3

NUM_EPOCHS = 10


@contexttimer.timer()
def run_experiments_baseline(
    config: DictConfig,
    data_helper: data_utils.TFFDataHelper,
) -> Dict:

    (_,
     server_optim_config,
     sample_config,
     _) = config_types.make_configs(
        config=config,
        model_index=data_helper.model_index,
        num_samples=config.tasks.num_samples,
        sample_max_norm=config.tasks.sample_max_norm,
        optim_learning_rate=config.tasks.optim_learning_rate,
        sample_learning_rate=config.tasks.sample_learning_rate)

    solver = algorithms_3.create_fed_avg(
        server_optim_config=server_optim_config,
        sample_config=sample_config,
        weight_decay=config.tasks.weight_decay,
    )

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    init_state = data_helper.get_client_train_objective(0).init(subkey)
    key, subkey = jax.random.split(key)
    trajectories, _ = solver(
        data_helper=data_helper,
        init_state=init_state,
        num_rounds=config.tasks.num_iterations,
        num_clients_per_round=config.tasks.num_clients_per_round,
        prng_key=subkey,
    )
    return trajectories


@contexttimer.timer()
def run_experiments_ep(
    config: DictConfig,
    data_helper: data_utils.TFFDataHelper,
    num_clients_per_round: int,
) -> Dict:

    if config.algorithm_name not in ["ep", "sep", "adf"]:
        raise ValueError(f"Unknown algorithm_name: {config.algorithm_name}")

    # -------------------------------------------------------------------------
    # Setting things up
    # -------------------------------------------------------------------------
    (client_optim_config,
     server_optim_config,
     sample_config,
     moment_config) = config_types.make_configs(
        config=config,
        model_index=data_helper.model_index,
        num_samples=config.tasks.num_samples,
        sample_max_norm=config.tasks.sample_max_norm,
        optim_learning_rate=config.tasks.optim_learning_rate,
        sample_learning_rate=config.tasks.sample_learning_rate)

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    initial_client_Sigma_scale = 1. / (
        (moment_config.initial_Lambda_scale +
         config.tasks.weight_decay) *
        config.moment.scale)
    (initial_time_index,
     initial_server_dist,
     initial_client_dists,
     additional_initial_info) = file_utils.load_server_dist(
        prng_key=subkey,
        data_helper=data_helper,
        dataset_name=config.tasks.dataset_name,
        round_index=config.tasks.finetuning.round_index,
        Sigma_scale=initial_client_Sigma_scale,
        stateful_client=(config.algorithm_name == "ep"))
    client_agents, server_agent = gaussian_utils.prepare_agents(
        gtype=config.gtype,
        dim=data_helper.dim,
        client_optim_config=client_optim_config,
        server_optim_config=server_optim_config,
        num_clients=data_helper.num_train_clients,
        stateful_client=(config.algorithm_name == "ep"),
        shared_basis_Q=None,
        initial_server_dist=initial_server_dist,
        initial_client_dists=initial_client_dists,
    )

    # For StackOverflow LR (with AdaGrad optimizer), we also need to resume
    # the optimizer state, which seems to matter quite a bit.
    # if config.tasks.dataset_name == "stackoverflowlr":
    #     mu_state = additional_initial_info["mu_state"][0].sum_of_squares
    #     server_agent.state[0][0].sum_of_squares["eta"] = deepcopy(mu_state)
    #     server_agent.state[0][0].sum_of_squares["Lambda"] = deepcopy(mu_state)

    damping_fn = misc_utils.get_damping_fn(
        damping_name="1/K",
        K=num_clients_per_round)

    trajectories: List[Dict[str, Any]] = []
    click.echo(
        click.style("Training Statistics: \n", fg="green", bold=True) +
        f"\t Finetuning Starting at {initial_time_index}\n" +
        f"\t Server Lambda: {server_agent.dist.Lambda[:10]}\n" +
        f"\t Client[0] Lambda: {client_agents[0].dist.Lambda[:10] if client_agents[0] is not None else None}\n"
    )

    # -------------------------------------------------------------------------
    # Experiments
    # -------------------------------------------------------------------------
    for time_index in trange(initial_time_index, config.tasks.num_iterations):
        # samples_round = []
        cavity_dists_round = []
        tilted_dists_round = []
        client_deltas_round = []
        tilted_objectives_round = []

        key, subkey = jax.random.split(key)
        client_indices_round = algorithms.sample_clients_uniformly(
            prng_key=subkey,
            num_clients_total=data_helper.num_train_clients,
            num_clients_to_sample=num_clients_per_round)

        for client_index in client_indices_round:
            key, subkey = jax.random.split(key)
            client_objective = data_helper.get_client_train_objective(client_index)

            # cavity distributions
            if config.algorithm_name == "ep":
                client_agent = client_agents[client_index].to("gpu")
                cavity_dist = gaussian_utils.from_quotient(
                    config.gtype,
                    server_agent.dist,
                    client_agent.dist)
            if config.algorithm_name == "sep":
                cavity_dist = gaussian_utils.from_power(
                    config.gtype,
                    server_agent.dist,
                    power=1. - client_objective.num_points / data_helper.total_train_num_points)
            if config.algorithm_name == "adf":
                cavity_dist = server_agent.dist
            if config.algorithm_name == "baseline":
                cavity_dist = gaussian_utils.get_uniform_gaussian(
                    config.gtype,
                    d=data_helper.dim,
                    Q=None)

            # Adding weight decay to the loss function,
            # and we will not subtract this prior when
            # computing the client delta. This design
            # is effectively adding a prior to the local
            # likelihoods. This design makes adding weight
            # decay to EP and PA easier.
            tilted_prior = gaussian_utils.add_weight_decay(
                config.gtype,
                q=cavity_dist,
                # The weight decay value is set for average loss
                # but we use sum loss here, so scale it accordingly.
                c=config.tasks.weight_decay * client_objective.num_points,
                d=data_helper.dim,
                Q=None)

            # tilted distributions
            tilted_objective = gaussian_utils.create_tilted_objective(
                gtype=config.gtype,
                objective=client_objective,
                prior=tilted_prior,
                prior_strength=1.0)

            tilted_dist = inference_utils.approximate_inference(
                method=f"{config.sampler_name}-{config.gtype}",
                prng_key=subkey,
                objective=tilted_objective,
                init_theta=server_agent.dist.mu,
                sample_config=sample_config,
                moment_config=moment_config,
                shared_basis_Q=None)

            # new local distributions
            client_delta = gaussian_utils.from_quotient(
                config.gtype,
                tilted_dist,
                server_agent.dist,
            )
            if config.tasks.optim_max_norm is not None:
                client_delta = gaussian_utils.clip_by_natural_params_global_norm(
                    config.gtype,
                    client_delta,
                    max_norm=config.tasks.optim_max_norm,
                )
            # damping
            client_delta = gaussian_utils.from_power(
                config.gtype,
                client_delta,
                power=damping_fn(time_index),
            )

            if config.algorithm_name == "ep":
                # update client states
                client_agents[client_index] = client_agent.update(client_delta).to("cpu")

            # logging
            # samples_round.append(samples)
            # cavity_dists_round.append(cavity_dist)
            # tilted_dists_round.append(tilted_dist)
            client_deltas_round.append(client_delta)
            # tilted_objectives_round.append(tilted_objective)

        # Update the server state after all clients have updated.
        server_delta = gaussian_utils.from_products(
            config.gtype,
            client_deltas_round)
        server_agent = server_agent.update(server_delta)

        if time_index % 100 == 0:
            trajectories.append({
                "round_index": time_index,
                "server_agent": deepcopy(server_agent),
                # "client_agents": deepcopy(client_agents),
                # "samples_round": deepcopy(samples_round),
                # "cavity_dists_round": deepcopy(cavity_dists_round),
                # "tilted_dists_round": deepcopy(tilted_dists_round),
                # "client_deltas_round": deepcopy(client_deltas_round),
                # "tilted_objectives_round": deepcopy(tilted_objectives_round),
                "client_indices_round": deepcopy(client_indices_round),
            })

        wandb.log({
            "round_index": time_index,
            **evaluation_utils.evaluate_objectives(
                dist=server_agent.dist,
                objective_train=data_helper.centralized_train_objective,
                objective_eval=data_helper.centralized_test_objective),
            **evaluation_utils.diagnostics(
                server_agent=server_agent,
                server_delta=server_delta,
                # cavity_dists_round=cavity_dists_round,
                # tilted_dists_round=tilted_dists_round,
                client_deltas_round=client_deltas_round,
                # tilted_objectives_round=tilted_objectives_round,
                # samples_round=samples_round,
            ),
        })

    return {
        "trajectories": trajectories,
        "server_agent": deepcopy(server_agent),
    }


def run_experiment(config: DictConfig) -> Dict:

    # For some reason, we need to put this line here instead of top of
    # the file, to make it work in multi-processing settings.
    tf.config.set_visible_devices([], device_type="GPU")

    if config.tasks.dataset_name == "cifar100":
        data_helper = data_utils.TFFCIFAR100(
            num_epochs=NUM_EPOCHS,
            num_clients=config.tasks.num_clients)

    if config.tasks.dataset_name == "cifar100toy":
        data_helper = data_utils.TFFCIFAR100Toy(
            num_epochs=NUM_EPOCHS,
            num_clients=config.tasks.num_clients)

    if config.tasks.dataset_name == "emnist62":
        data_helper = data_utils.TFFEMNIST62(
            num_epochs=NUM_EPOCHS,
            num_clients=config.tasks.num_clients)

    if config.tasks.dataset_name == "stackoverflowlr":
        data_helper = data_utils.TFFStackOverflowLR(
            num_epochs=NUM_EPOCHS,
            num_clients=config.tasks.num_clients)

    if config.which_experiment == "ep":
        output = run_experiments_ep(
            config=config,
            data_helper=data_helper,
            num_clients_per_round=config.tasks.num_clients_per_round)

    if config.which_experiment == "baseline":
        output = run_experiments_baseline(
            config=config,
            data_helper=data_helper)

    outputs_dict = {
        "output": output,
        **OmegaConf.to_container(config),
    }

    if config.base_file_name is not None:
        file_name = f"{config.base_file_name}.cpkl"
        print(f"Saveing to {file_name}")
        misc_utils.dump(outputs_dict, file_name)

    return outputs_dict
