import attr
import jax
import optax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional

from ..utils import misc_utils
from ..objectives import gaussians


@attr.s(eq=False, order=False, frozen=True)
class Agent(object):
    rounds: int = attr.ib()
    dist: gaussians.Gaussian = attr.ib()
    state: optax.OptState = attr.ib()
    optimizer: optax.GradientTransformation = attr.ib()

    @staticmethod
    def dist_to_params(
            dist: gaussians.Gaussian,
        ) -> Tuple[Dict[str, jnp.ndarray],
                   Optional[Dict[str, jnp.ndarray]]]:
        params = {
            "eta": dist.eta,
            "Lambda": dist.Lambda,
        }
        frozen_params = None
        return params, frozen_params

    @staticmethod
    def params_to_dist(
            params: Dict[str, jnp.ndarray],
            frozen_params: Optional[Dict[str, jnp.ndarray]],
            frozen_params_delta: Optional[Dict[str, jnp.ndarray]],
    ) -> gaussians.Gaussian:
        return gaussians.Gaussian(
            eta=params["eta"],
            Lambda=params["Lambda"],
        )

    @classmethod
    def from_dist(
            cls,
            dist: gaussians.Gaussian,
            optimizer: optax.GradientTransformation,
    ) -> "Agent":
        # https://github.com/deepmind/optax/blob/master/examples/quick_start.ipynb
        # **Important.**
        # We would be __adding__ the deltas to the params. Hence,
        # we need to use __negative__ learning rate.
        # Note that inside the optimizer, they already flip the sign
        # because internally the optimizer adds update. So we are
        # essentially flipping it again.
        params, _ = cls.dist_to_params(dist)
        optimizer = optax.chain(
            optimizer,
            optax.scale(-1.))
        state = optimizer.init(params)

        return cls(
            rounds=0,
            dist=dist,
            state=state,
            optimizer=optimizer)

    def update(self, delta: gaussians.Gaussian) -> "Agent":
        # Preprocessing
        (params,
         frozen_params) = (
            self.dist_to_params(self.dist))
        (params_delta,
         frozen_params_delta) = (
            self.dist_to_params(delta))
        # Updates
        updates, new_state = self.optimizer.update(
            params_delta, self.state)
        params = optax.apply_updates(params, updates)
        # Postprocessing
        new_dist = self.params_to_dist(
            params,
            frozen_params,
            frozen_params_delta)

        return type(self)(
            rounds=self.rounds + 1,
            dist=new_dist,
            state=new_state,
            optimizer=self.optimizer,
        )

    def to(self, device_name: str) -> "Agent":

        def _state_to(state: optax.OptState) -> optax.OptState:
            device = misc_utils.get_device(device_name)
            return jax.device_put(state, device)

        return type(self)(
            rounds=self.rounds,
            dist=self.dist.to(device_name),
            state=_state_to(self.state),
            optimizer=self.optimizer,
        )


@attr.s(eq=False, order=False, frozen=True)
class DiagonalAgent(Agent):

    @staticmethod
    def params_to_dist(
            params: Dict[str, jnp.ndarray],
            frozen_params: Optional[Dict[str, jnp.ndarray]],
            frozen_params_delta: Optional[Dict[str, jnp.ndarray]],
    ) -> gaussians.DiagonalGaussian:
        return gaussians.DiagonalGaussian(
            eta=params["eta"],
            Lambda=params["Lambda"],
        )
