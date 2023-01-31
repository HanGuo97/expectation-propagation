import attr
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Tuple, Optional
from ..objectives.base import StochasticObjective


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
    """Represents the server state."""

    r: int = attr.ib()
    x: jnp.ndarray = attr.ib()
    v: jnp.ndarray = attr.ib()


# Type aliases.
ClientUpdateFn = Callable[
    [
        StochasticObjective,  # client objective function
        jnp.ndarray,  # initial state
    ],
    jnp.ndarray,  # client delta
]
ServerUpdateFn = Callable[
    [
        List[jnp.ndarray],  # client deltas
        jnp.ndarray,  # client weights
        ServerState,  # previous server state
    ],
    ServerState,  # updated server state
]
SampleClientsFn = Callable[
    [
        jnp.ndarray,  # prng key
        int,  # the total number of clients
        int,  # the number of clients to sample
    ],
    jnp.ndarray,  # sampled client ids
]
RoundInfo = Dict[str, Any]
FederatedLearningFn = Callable[
    [
        List[StochasticObjective],  # a list of client objectives
        jnp.ndarray,  # initial state
        jnp.ndarray,  # prng key
        int,  # number of round
        int,  # number of clients per round
    ],
    Tuple[List[ServerState], List[RoundInfo]],  # trajectory
]


@attr.s(eq=False, order=False, frozen=True)
class ServerState2(object):
    r: int = attr.ib()
    # the global mean
    mu_global: jnp.ndarray = attr.ib()
    # the global sigma
    sigma_global: jnp.ndarray = attr.ib()
    # Additional information.
    additional_info: Optional[Dict[str, Any]] = attr.ib(default=None)


@attr.s(eq=False, order=False, frozen=True)
class ClientState(object):
    r: int = attr.ib()
    # the local mean
    mu_local: jnp.ndarray = attr.ib()
    # the local sigma
    sigma_local: jnp.ndarray = attr.ib()
    # Additional information.
    additional_info: Optional[Dict[str, Any]] = attr.ib(default=None)


@attr.s(eq=False, order=False, frozen=True)
class ServerMessage(object):
    # the global mean
    mu_global: jnp.ndarray = attr.ib()
    # the global sigma
    sigma_global: jnp.ndarray = attr.ib()


@attr.s(eq=False, order=False, frozen=True)
class ClientMessage(object):
    # the local mu
    mu_local: jnp.ndarray = attr.ib()
    # the local sigma
    sigma_local: jnp.ndarray = attr.ib()


ClientUpdateFn2 = Callable[
    [
        StochasticObjective,  # client objective function
        ClientState,  # previous client state
        ServerMessage,  # server message
        jnp.ndarray,  # prng key
    ],
    Tuple[
        ClientMessage,
        ClientState,
    ],
]
ServerUpdateFn2 = Callable[
    [
        List[ClientMessage],  # client messages
        jnp.ndarray,  # client weights
        ServerState,  # previous server state
    ],
    Tuple[
        ServerMessage,
        ServerState2,
    ],
]
