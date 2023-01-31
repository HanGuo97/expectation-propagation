import jax
import jax.numpy as jnp
import click
from typing import Tuple, List, Dict, Optional

from . import data_utils
from . import misc_utils
from . import gaussian_utils
from ..objectives import gaussians
from ..utils.types import ServerState

FILE_NAMES = {
    "cifar100": None,
    "emnist62": None,
    "stackoverflowlr": None,
}


def load_server_dist(
    data_helper: data_utils.TFFDataHelper,
    dataset_name: str,
    Sigma_scale: float,
    prng_key: jax.random.KeyArray,
    round_index: Optional[int] = None,
    stateful_client: bool = False,
) -> Tuple[int, gaussians.Gaussian, List[Optional[gaussians.Gaussian]], Optional[Dict]]:
    if round_index is not None:
        trajectory_index = int(round_index / 100) + 1
        trajectories = misc_utils.load(FILE_NAMES[dataset_name])
        trajectory: ServerState = trajectories["output"][trajectory_index]
        mu = trajectory.x
        starting_round_index = trajectory.r
        additional_info = {"mu_state": trajectory.v}
        click.secho("Loaded.", fg="green")
    else:
        # Init function requires some sample data to infer
        # the shapes, so we just pick a random client.
        mu = data_helper.get_client_train_objective(0).init(prng_key)
        starting_round_index = 0
        additional_info = None
        click.secho("From Scratch.", fg="green")

    if stateful_client is True:
        client_dists = []
        for k in range(data_helper.num_train_clients):
            client_Sigma = (
                Sigma_scale *
                jnp.ones_like(mu) /
                data_helper.get_client_train_num_points(k))
            client_dist = gaussian_utils.from_mu_and_Sigma(
                "diagonal", mu=mu, Sigma=client_Sigma).to("cpu")
            client_dists.append(client_dist)
    else:
        client_dists = [
            None for k in
            range(data_helper.num_train_clients)]

    server_Sigma = (
        Sigma_scale *
        jnp.ones_like(mu) /
        data_helper.total_train_num_points)
    server_dist = gaussian_utils.from_mu_and_Sigma(
        "diagonal", mu=mu, Sigma=server_Sigma)

    return starting_round_index, server_dist, client_dists, additional_info
