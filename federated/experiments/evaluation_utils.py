import jax
import jax.numpy as jnp
import pandas as pd
from tqdm.auto import tqdm
from matplotlib.axes import SubplotBase
from typing import List, Dict, Optional, Any

from .misc_utils import running_mean
from ..modules.utils import ModelIndex
from ..experiments.agent_utils import Agent
from ..objectives.gaussians import (
    Gaussian,
    DiagonalGaussian)
from ..objectives.logistics_regression import (
    SimpleObjective,
    ObjectiveWithDiagGaussianPrior)


def _vector_statistics(x: jnp.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}/element-max": x.max(),
        f"{prefix}/element-min": x.min(),
        f"{prefix}/element-mean": x.mean(),
        f"{prefix}/2-norm": jnp.linalg.norm(x, ord=2),
        f"{prefix}/inf-norm": jnp.linalg.norm(x, ord=jnp.inf),
        f"{prefix}/-inf-norm": jnp.linalg.norm(x, ord=-jnp.inf),
    }


def diagnostics(
    server_agent: Agent,
    server_delta: Optional[Gaussian] = None,
    cavity_dists_round: Optional[List[Gaussian]] = None,
    tilted_dists_round: Optional[List[Gaussian]] = None,
    client_deltas_round: Optional[List[Gaussian]] = None,
    tilted_objectives_round: Optional[List[SimpleObjective]] = None,
    samples_round: Optional[List[jnp.ndarray]] = None,
) -> Dict[str, Any]:

    if isinstance(server_agent, DiagonalGaussian):
        raise TypeError

    info = {
        **_vector_statistics(
            server_agent.dist.Lambda,
            prefix="server/Lambda"),
        **_vector_statistics(
            server_agent.dist.Sigma,
            prefix="server/Sigma"),
        **_vector_statistics(
            server_agent.dist.eta,
            prefix="server/eta"),
        **_vector_statistics(
            server_agent.dist.mu,
            prefix="server/mu"),
    }

    if server_delta is not None:
        info.update({
            **_vector_statistics(
                server_delta.Lambda,
                prefix=f"server_delta/Lambda"),
            **_vector_statistics(
                server_delta.eta,
                prefix=f"server_delta/eta"),
        })

    if client_deltas_round is not None:
        client_deltas_Lambda_norms = jnp.stack([
            jnp.linalg.norm(client_delta.Lambda, ord=2)
            for client_delta in client_deltas_round],
            axis=0)
        client_deltas_eta_norms = jnp.stack([
            jnp.linalg.norm(client_delta.eta, ord=2)
            for client_delta in client_deltas_round],
            axis=0)

        info.update({
            "client_deltas/Lambda/2-norm/mean": client_deltas_Lambda_norms.mean(),
            "client_deltas/Lambda/2-norm/max": client_deltas_Lambda_norms.max(),
            "client_deltas/Lambda/2-norm/min": client_deltas_Lambda_norms.min(),
            "client_deltas/eta/2-norm/mean": client_deltas_eta_norms.mean(),
            "client_deltas/eta/2-norm/max": client_deltas_eta_norms.max(),
            "client_deltas/eta/2-norm/min": client_deltas_eta_norms.min(),
        })

    return info


def evaluate_objectives(
    dist: Gaussian,
    objective_train: SimpleObjective,
    objective_eval: SimpleObjective,
) -> Dict[str, float]:

    train_metrics = objective_train.evaluate(
        dist.mu,
        prng_key=jax.random.PRNGKey(0))

    if objective_eval.model_index == ModelIndex.EMNIST62:
        eval_metrics = objective_eval.evaluate_in_batches(
            dist.mu,
            prng_key=jax.random.PRNGKey(0),
            cutoff=40000)
    else:
        eval_metrics = objective_eval.evaluate(
            dist.mu,
            prng_key=jax.random.PRNGKey(0))

    return {
        **dict(
            (f"train_{k}", v) for k, v
            in train_metrics.items()),
        **dict(
            (f"eval_{k}", v) for k, v
            in eval_metrics.items()),
    }


def plot_smooth(
        axis: SubplotBase,
        Xs: List,
        Ys: List,
        label: Optional[str],
        alphas: Optional[List[float]],
        window: int,
        *args, **kwargs
) -> None:

    if alphas is None:
        alphas = [0.02, 0.9]

    axis.plot(
        Xs,
        Ys,
        *args,
        **kwargs,
        alpha=alphas[0])
    axis.plot(
        Xs,
        running_mean(Ys, window=window),
        *args,
        **kwargs,
        label=label,
        alpha=alphas[1])
