import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


class HackyGroupNorm(hk.Module):
  def __init__(
        self,
        create_scale : bool,
        create_offset: bool,
        decay_rate: float,
        eps: float,
        name: Optional[str] = None,
    ):
        # We do not need these.
        del create_scale, create_offset, decay_rate, eps

        super().__init__(name=name)
        # https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/simulation/models/group_norm.py
        self._norm = hk.GroupNorm(
            groups=2,
            axis=3,
            create_scale=False,
            create_offset=False,
            eps=1e-3,
        )

  def __call__(
        self, x: jnp.ndarray,
        is_training: bool,
        test_local_stats: bool,
    ):
        # Again, we do not need these.
        del is_training, test_local_stats

        return self._norm(x)

