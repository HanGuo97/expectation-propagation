import jax
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from sklearn.datasets import make_spd_matrix
from typing import List, Tuple, Union, Dict, Optional, Callable

from . import misc_utils
from . import agent_utils
from ..utils.misc_utils import get_device
from ..objectives import gaussians
from ..objectives import logistics_regression
from ..inference import optimization_utils
from ..utils import config_types


def from_mu_and_Sigma(
        gtype: str,
        mu: jnp.ndarray,
        Sigma: gaussians.Covariance,
        Q: Optional[jnp.ndarray] = None,
) -> gaussians.Gaussian:

    if gtype == "standard":
        return gaussians.SimpleGaussian(
            mu=mu,
            Sigma=Sigma)

    if gtype == "diagonal":
        return gaussians.DiagonalGaussian(
            mu=mu,
            Sigma=Sigma)

    raise ValueError


def get_agent(gtype: str, *args, **kwargs) -> agent_utils.Agent:

    if gtype == "standard":
        return agent_utils.Agent.from_dist(*args, **kwargs)

    if gtype == "diagonal":
        return agent_utils.DiagonalAgent.from_dist(*args, **kwargs)

    raise ValueError


def prepare_agents(
    gtype: str,
    dim: int,
    client_optim_config: config_types.OptimizerConfig,
    server_optim_config: config_types.OptimizerConfig,
    num_clients: int,
    stateful_client: bool = True,
    shared_basis_Q: Optional[jnp.ndarray] = None,
    initial_server_dist: Optional[gaussians.Gaussian] = None,
    initial_client_dists: Optional[List[gaussians.Gaussian]] = None,
) -> Tuple[List[Optional[agent_utils.Agent]],
           agent_utils.Agent]:

    client_optimizer = optimization_utils.create_optimizer(
        name=client_optim_config.name,
        learning_rate=client_optim_config.learning_rate,
        max_norm=client_optim_config.max_norm,
        **client_optim_config.kwargs)

    server_optimizer = optimization_utils.create_optimizer(
        name=server_optim_config.name,
        learning_rate=server_optim_config.learning_rate,
        max_norm=server_optim_config.max_norm,
        **server_optim_config.kwargs)

    if initial_server_dist is None:
        server_dist = get_uniform_gaussian(
            gtype,
            d=dim,
            Q=shared_basis_Q)
    else:
        server_dist = initial_server_dist

    if stateful_client is True:
        if initial_client_dists is None:
            client_dists = [
                from_power(
                    gtype,
                    server_dist,
                    power=1. / num_clients).to("cpu")
                for _ in range(num_clients)]
        else:
            client_dists = [
                q.to("cpu") for q
                in initial_client_dists]

        client_agents = [
            get_agent(
                gtype,
                dist=_client_dist,
                optimizer=client_optimizer).to("cpu")
            for _client_dist in client_dists]
    else:
        client_agents = [None for _ in range(num_clients)]

    server_agent = get_agent(
        gtype,
        dist=server_dist,
        optimizer=server_optimizer).to("gpu")

    return client_agents, server_agent


def get_uniform_gaussian(
        gtype: str,
        d: int,
        Q: Optional[jnp.ndarray] = None,
        device_name: Optional[str] = None,
    ) -> gaussians.Gaussian:

    def _to(x: jnp.ndarray) -> jnp.ndarray:
        device = get_device(device_name)
        return jax.device_put(x, device)

    if gtype == "standard":
        return gaussians.SimpleGaussian(
            eta=   _to(jnp.zeros([d])),
            Lambda=_to(jnp.zeros([d, d])),
        )

    if gtype == "diagonal":
        return gaussians.DiagonalGaussian(
            eta=   _to(jnp.zeros([d])),
            Lambda=_to(jnp.zeros([d])),
        )

    raise ValueError


def get_unit_gaussian(
        gtype: str,
        d: int,
        Q: Optional[jnp.ndarray] = None,
        device_name: Optional[str] = None,
) -> gaussians.Gaussian:

    def _to(x: jnp.ndarray) -> jnp.ndarray:
        device = get_device(device_name)
        return jax.device_put(x, device)

    if gtype == "standard":
        return gaussians.SimpleGaussian(
            eta=   _to(jnp.zeros([d])),
            Lambda=_to(jnp.eye(d)),
        )

    if gtype == "diagonal":
        return gaussians.DiagonalGaussian(
            eta=   _to(jnp.zeros([d])),
            Lambda=_to(jnp.ones(d)),
        )

    raise ValueError


def get_random_gaussian(
        gtype: str,
        d: int,
        Q: Optional[jnp.ndarray] = None,
        device_name: Optional[str] = None,
) -> gaussians.Gaussian:

    def _to(x: jnp.ndarray) -> jnp.ndarray:
        device = get_device(device_name)
        return jax.device_put(x, device)

    if gtype == "standard":
        return gaussians.SimpleGaussian(
            eta=jnp.array(np.random.normal(size=[d])),
            Lambda=jnp.array(make_spd_matrix(n_dim=d)),
        )

    if gtype == "diagonal":
        return gaussians.DiagonalGaussian(
            eta=jnp.array(np.random.normal(size=[d])),
            # To make sure the covariance matrix is positive definite,
            Lambda=jnp.array(np.random.normal(size=[d])) ** 2 + 1e-7,
        )

    raise ValueError


def add_weight_decay(
        gtype: str,
        q: gaussians.Gaussian,
        c: float,
        d: int,
        Q: Optional[jnp.ndarray] = None,
        device_name: Optional[str] = None,
) -> gaussians.Gaussian:

    q0 = get_unit_gaussian(
        gtype,
        d=d,
        Q=Q,
        device_name=device_name)

    q0 = from_power(
        gtype,
        q=q0,
        power=c)

    return from_products(gtype, [q, q0])


def from_products(gtype: str, *args, **kwargs) -> gaussians.Gaussian:

    if gtype == "standard":
        return gaussians.SimpleGaussian.from_products(*args, **kwargs)

    if gtype == "diagonal":
        return gaussians.DiagonalGaussian.from_products(*args, **kwargs)

    raise ValueError


def from_quotient(gtype: str, *args, **kwargs) -> gaussians.Gaussian:

    if gtype == "standard":
        return gaussians.SimpleGaussian.from_quotient(*args, **kwargs)

    if gtype == "diagonal":
        return gaussians.DiagonalGaussian.from_quotient(*args, **kwargs)

    raise ValueError


def from_power(gtype: str, *args, **kwargs) -> gaussians.Gaussian:

    if gtype == "standard":
        return gaussians.SimpleGaussian.from_power(*args, **kwargs)

    if gtype == "diagonal":
        return gaussians.DiagonalGaussian.from_power(*args, **kwargs)

    raise ValueError


def clip_by_natural_params_global_norm(
    gtype: str,
    q: gaussians.Gaussian,
    max_norm: float,
) -> gaussians.Gaussian:

    if gtype == "diagonal":
        params = {
            "eta": q.eta,
            "Lambda": q.Lambda,
        }
        params = optimization_utils.clip_by_global_norm(
            params=params,
            max_norm=max_norm,
        )
        return gaussians.DiagonalGaussian(
            eta=params["eta"],
            Lambda=params["Lambda"],
        )

    raise ValueError


def create_tilted_objective(
        gtype: str,
        objective: logistics_regression.SimpleObjective,
        prior: gaussians.Gaussian,
        prior_strength: float
) -> logistics_regression.SimpleObjective:

    if gtype == "standard":
        return logistics_regression.ObjectiveWithGaussianPrior.from_objective(
            objective=objective,
            prior=prior,
            prior_strength=prior_strength)

    if gtype == "diagonal":
        return logistics_regression.ObjectiveWithDiagGaussianPrior.from_objective(
            objective=objective,
            prior=prior,
            prior_strength=prior_strength)

    raise ValueError
