import attr
import jax.numpy as jnp
from omegaconf import DictConfig
from typing import Dict, Optional, Any, Tuple

from ..modules.utils import ModelIndex


@attr.s(eq=False, order=False, frozen=True)
class OptimizerConfig(object):
    name: str = attr.ib()
    learning_rate: float = attr.ib()
    max_norm: Optional[float] = attr.ib()
    kwargs: Dict[str, Any] = attr.ib()


@attr.s(eq=False, order=False, frozen=True)
class SampleOptimizationConfig(object):
    num_samples: int = attr.ib()
    optim_config: OptimizerConfig = attr.ib()
    num_chains: int = attr.ib(default=1)
    thinning: int = attr.ib(default=1)


@attr.s(eq=False, order=False, frozen=True)
class MomentConfig(object):
    method: str = attr.ib()
    scale: float = attr.ib()
    shrinkage: float = attr.ib()
    num_epochs: int = attr.ib()
    num_samples: int = attr.ib()
    weight_decay: float = attr.ib()
    clip_ratio: float = attr.ib()
    mu_learning_rate: float = attr.ib()
    Lambda_decay_rate: float = attr.ib()
    initial_Lambda_scale: float = attr.ib()


def make_configs(
    config: DictConfig,
    model_index: ModelIndex,
    num_samples: int,
    sample_max_norm: Optional[float],
    optim_learning_rate: float,
    sample_learning_rate: float,
) -> Tuple[
    OptimizerConfig,
    OptimizerConfig,
    SampleOptimizationConfig,
    MomentConfig,
]:

    if model_index in [ModelIndex.CIFAR100, ModelIndex.EMNIST62]:
        client_optim_config = OptimizerConfig(
            name="sgd",
            learning_rate=optim_learning_rate,
            max_norm=None,
            kwargs={"momentum": 0.9},
        )
        server_optim_config = OptimizerConfig(
            name="sgd",
            learning_rate=optim_learning_rate,
            max_norm=None,
            kwargs={"momentum": 0.9},
        )

    if model_index in [ModelIndex.CIFAR100_TOY]:
        client_optim_config = OptimizerConfig(
            name="sgd",
            learning_rate=optim_learning_rate,
            max_norm=None,
            kwargs={},
        )
        server_optim_config = OptimizerConfig(
            name="sgd",
            learning_rate=optim_learning_rate,
            max_norm=None,
            kwargs={},
        )

    if model_index in [ModelIndex.STACKOVERFLOW_LR]:
        client_optim_config = OptimizerConfig(
            name="sgd",
            learning_rate=optim_learning_rate,
            max_norm=None,
            kwargs={},
        )
        server_optim_config = OptimizerConfig(
            name="adagrad",
            learning_rate=optim_learning_rate,
            max_norm=None,
            kwargs={"initial_accumulator_value": 0.0, "eps": 1e-5},
        )

    sample_config = SampleOptimizationConfig(
        num_samples=num_samples,
        optim_config=OptimizerConfig(
            name="sgd",
            learning_rate=sample_learning_rate,
            max_norm=sample_max_norm,
            kwargs={"momentum": 0.9},
        ),
    )

    moment_config = MomentConfig(
        method=config.moment.method,
        scale=config.moment.scale,
        shrinkage=config.moment.shrinkage,
        num_epochs=config.moment.num_epochs,
        num_samples=config.moment.num_samples,
        weight_decay=config.tasks.weight_decay,  # this is not in `moment`
        clip_ratio=config.moment.clip_ratio,
        mu_learning_rate=config.moment.mu_learning_rate,
        Lambda_decay_rate=config.moment.Lambda_decay_rate,
        initial_Lambda_scale=config.moment.initial_Lambda_scale,
    )

    return (
        client_optim_config,
        server_optim_config,
        sample_config,
        moment_config,
    )
