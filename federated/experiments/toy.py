import jax
import imageio
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from glob import glob
from typing import List, cast
from scipy.spatial import distance
from sklearn.datasets import make_spd_matrix
from scipy.stats import invwishart, multivariate_normal

from .gaussian_utils import (
    get_uniform_gaussian,
    get_random_gaussian,
    from_mu_and_Sigma)
from ..objectives.gaussians import SimpleGaussian

DIM = 2


def initialize(num_clients: int) -> List[SimpleGaussian]:
    return [
        cast(
            SimpleGaussian,
            get_uniform_gaussian("standard", d=DIM))
        for k in range(num_clients)
    ]


def approximate_diagonal(p: SimpleGaussian) -> SimpleGaussian:
    """With moment matching, the best diagonal approximation is the
       Gaussian with the same mean, but the Covariance matrix contains
       only the diagonal entries of the original matrix.
    """
    return SimpleGaussian(
        mu=p.mu,
        Sigma=jnp.diagflat(jnp.diagonal(p.Sigma)))


def approximate_identity(p: SimpleGaussian, p0: SimpleGaussian) -> SimpleGaussian:
    Lambda = jnp.diagflat(jnp.diagonal(p0.Lambda) + 1.)
    return SimpleGaussian(
        mu=p.mu,
        Sigma=jnp.linalg.inv(Lambda))


def step(
    p_clients: List[SimpleGaussian],
    q_clients: List[SimpleGaussian],
    method: str,
) -> List[SimpleGaussian]:
    q_global = SimpleGaussian.from_products(q_clients)
    for k in range(len(q_clients)):
        q_cavity = SimpleGaussian.from_quotient(q_global, q_clients[k])
        q_tilted = SimpleGaussian.from_products([q_cavity, p_clients[k]])
        if method == "diagonal":
            q_tilted = approximate_diagonal(q_tilted)
        if method == "identity":
            q_tilted = approximate_identity(q_tilted, q_cavity)
        q_clients[k] = SimpleGaussian.from_quotient(q_tilted, q_cavity)

    return q_clients


def run_experiment(method: str, base_file_name: str) -> None:

    Xs, Ys = np.mgrid[-10:15:.01, -15:7.5:.01]
    XsYs = np.dstack((Xs, Ys))

    def _plot_Gaussian(_q: SimpleGaussian, axis: plt.Axes, alpha: float, cmap_name: str) -> None:
        axis.contourf(
            Xs,
            Ys,
            multivariate_normal(
                _q.mu,
                _q.Sigma).pdf(XsYs),
            levels=5,
            alpha=alpha,
            cmap=plt.get_cmap(cmap_name))


    # mus = [
    #     np.array([3.5, -2.3]) * 2,
    #     np.array([3.1, 1.12]) * 2,
    # ]
    # Sigmas = [
    #     np.array([
    #         [3.0, 2.3],
    #         [2.3, 2.5]]),
    #     np.array([
    #         [5.0, 2.3],
    #         [2.3, 1.5]]),
    # ]
    mus = [
        np.array([3.5, -5.3]) * 2,
        np.array([0.1, 1.12]) * 2,
    ]
    Sigmas = [
        np.array([
            [11.0, -5.3],
            [-5.3, 3.5]]),
        np.array([
            [5.0, 4.9],
            [4.9, 5.5]]),
    ]
    mu_fedavg = np.stack(mus, axis=0).mean(axis=0)

    p_clients = [
        SimpleGaussian(
            mu=jnp.array(mu),
            Sigma=jnp.array(Sigma),
        )
        for mu, Sigma in zip(mus, Sigmas)
    ]
    p_global = SimpleGaussian.from_products(p_clients)

    # Experiment
    mus_global = []
    q_clients = initialize(len(p_clients))
    for t in range(10):
        q_clients = step(p_clients, q_clients, method=method)
        q_global = SimpleGaussian.from_products(q_clients)
        mus_global.append(q_global.mu)
    mus_global_stacked = np.stack(mus_global, axis=0)

    fig, axis = plt.subplots(
        nrows=1,
        ncols=1,
        dpi=100,
        figsize=(5, 5))

    _plot_Gaussian(
        p_global,
        axis=axis,
        alpha=1.0,
        cmap_name="Greys")
    for _p in p_clients:
        _plot_Gaussian(
            _p,
            axis=axis,
            alpha=0.25,
            cmap_name="Greys")

    axis.plot(
        mu_fedavg[0],
        mu_fedavg[1],
        color="#403d39",
        marker="*",
        markersize=15,
    )
    axis.plot(
        mus_global[0][0],
        mus_global[0][1],
        color="#197278",
        marker="*",
        markersize=15,
    )
    axis.plot(
        mus_global[-1][0],
        mus_global[-1][1],
        color="#bc4749",
        marker="*",
        markersize=15,
    )
    axis.plot(
        mus_global_stacked[:, 0],
        mus_global_stacked[:, 1],
        linestyle="dashed",
        marker="o",
        markersize=5,
        color="#9c6644",
        alpha=0.5,
    )
    if base_file_name is not None:
        fig.savefig(f"{base_file_name}.pdf")


def run_experiment_2(method: str, num_steps: int = 100) -> List[float]:
    p_clients = [generate_random_client() for _ in range(2)]
    p_global = SimpleGaussian.from_products(p_clients)

    # Experiment
    errors = []
    q_clients = initialize(len(p_clients))
    for t in range(num_steps):
        q_clients = step(p_clients, q_clients, method=method)
        q_global = SimpleGaussian.from_products(q_clients)
        error = distance.euclidean(p_global.mu, q_global.mu)
        errors.append(error)

    return errors


def generate_random_client(lam: float = 0.2) -> SimpleGaussian:
    "Normal-inverse-Wishart distribution"
    Scale = make_spd_matrix(2, random_state=0)
    Sigma = invwishart.rvs(df=7, scale=Scale)
    Sigma = Sigma / lam
    mu = multivariate_normal.rvs(cov=Sigma)
    return from_mu_and_Sigma("standard", mu=mu, Sigma=Sigma)